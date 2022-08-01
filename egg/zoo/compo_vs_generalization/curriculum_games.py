import torch
import torch.nn as nn
import scipy.special
from abc import abstractmethod
from egg.core.reinforce_wrappers import SenderReceiverRnnReinforce
from egg.zoo.compo_vs_generalization.data import floatable


class CurriculumGameWrapper(nn.Module):
    """
    Abstract game wrapper for games that implement a curriculum.
    Child classes should have a curriculum_level attribute
    as well as implement an update_curriculum_level method.
    """

    def __init__(self, game: SenderReceiverRnnReinforce):
        super().__init__()
        self.game = game

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        return self.game(sender_input, labels, receiver_input=None, aux_input=None)

    @abstractmethod
    def update_curriculum_level(self):
        """
        This method is called by the trainer
        and should schedule the game's curriculum
        """
        pass

    def __getattr__(self, attrname):
        """
        Lifting attributes :
        Attributes of wrapper.game can be accessed as if
        they were attributes of wrapper
        """
        try:
            return super().__getattr__(attrname)
        except AttributeError:
            return getattr(self.game, attrname)


class GraduallyRevealAttributes(CurriculumGameWrapper):
    """
    In this game wrapper, all attributes are masked except {n_revealed} of them.
    The position of unmasked attributes can be either random if {mode} is
    'random', or at the left if {mode} is 'from_left_to_right'.
    During training, more attributes are gradually revealed as n_revealed
    augments.
    """

    def __init__(self,
                 game: SenderReceiverRnnReinforce,
                 n_attributes: int,
                 n_values: int,
                 mask_positioning: str,
                 masking_mode: str,
                 reveal_distribution: str,
                 initial_n_unmasked: int,
                 ):
        valid_mask_positionings = ['left_to_right', 'random']
        if mask_positioning not in valid_mask_positionings:
            raise ValueError(
                f"Invalid mask_positioning {mask_positioning}. mode should be in {valid_mask_positionings}")

        valid_masking_modes = ['zero_out', 'dedicated_value']
        if masking_mode not in valid_masking_modes:
            raise ValueError(f"Invalid masking_mode {masking_mode}. mode should be in {valid_masking_modes}")

        valid_reveal_distribution = ['deterministic', 'uniform', 'natural', 'binomial']
        if reveal_distribution not in valid_reveal_distribution and not floatable(reveal_distribution):
            raise ValueError(
                f"Invalid reveal distribution {reveal_distribution}. mode should be in {valid_reveal_distribution}")

        super().__init__(game)
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.mask_positioning = mask_positioning
        self.masking_mode = masking_mode
        self.reveal_distribution = reveal_distribution
        self.curriculum_level = min(initial_n_unmasked, n_attributes)

        # number of attributes to reveal when testing (will be set by callback)
        self.n_revealed_test = None

        # last accuracy for curriculum purposes
        self.last_acc = 0.

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        batch_size = sender_input.shape[0]

        sender_input, mask = self.mask_attributes(sender_input)

        # pass masking info to loss (for logging)
        if aux_input is None:
            aux_input = {}
        aux_input['mask'] = mask

        optimized_loss, interaction = self.game(
            sender_input,
            labels,
            receiver_input=None,
            aux_input=aux_input
        )
        interaction.aux['curriculum_level'] = torch.tensor(
            self.curriculum_level,
            device=sender_input.device,
            dtype=torch.float32).expand(
            batch_size,
            1)
        return optimized_loss, interaction

    def mask_attributes(self, sender_input):
        """
        Takes in a batch of inputs and mask attributes according to the game's internal
        parameters. returns the masked batch and the mask
        """
        batch_size = sender_input.shape[0]
        device = sender_input.device

        # n_revealed specifies the number of revealed attributes for each
        # element of the batch. It is of shape (batch_size,)
        if self.reveal_distribution == 'deterministic':
            probs = torch.zeros((batch_size, self.curriculum_level))
            probs[:, -1] = 1.
        elif self.reveal_distribution == 'uniform':
            probs = torch.ones((batch_size, self.curriculum_level)) / self.curriculum_level
        elif self.reveal_distribution == 'natural':
            # this distribution simulates sampling from a dataset containing
            # every masked input (up to curriculum_level masks).
            # For k masks, there are (n_attributes choose k)*n_values^k elements

            probs = torch.tensor(
                [scipy.special.comb(self.n_attributes, k) * self.n_values**k
                 for k in range(1, self.curriculum_level + 1)]
            )
            probs = probs.expand(batch_size, self.curriculum_level)
        elif floatable(self.reveal_distribution):
            # if reveal distribution is a float, we treat it as a temperature parameter
            # and create a distribution between deterministic and uniform
            probs = torch.zeros((batch_size, self.curriculum_level))
            probs[:, -1] = 1./float(self.reveal_distribution)
            probs = torch.softmax(probs, dim=1)
        elif self.reveal_distribution == 'binomial':
            # binomial distribution, with a sliding mean based on last accuracy.
            # the distribution is interpolated with the uniform distribution
            # through a parameter t in 0, 1
            t = 0.3
            probs = torch.distributions.binomial.Binomial(self.curriculum_level-1, min(self.last_acc, 0.95))
            probs = torch.exp(probs.log_prob(probs.enumerate_support()))  # get actual probability mass function
            probs = t/self.curriculum_level + (1-t)*probs  # interpolation with uniform distribution
            probs = probs.expand(batch_size, self.curriculum_level)



        else:
            raise NotImplemented

        if self.training or self.n_revealed_test is None:
            n_revealed = torch.multinomial(probs, 1).to(device)
            n_revealed += 1  # so that n_revealed is between 1 and n_attributes
        else:
            n_revealed = torch.zeros((batch_size,)).to(device) + self.n_revealed_test

        if self.mask_positioning == 'left_to_right':
            idxs_to_reveal = torch.arange(self.n_attributes, dtype=torch.long)
            idxs_to_reveal = idxs_to_reveal.expand(batch_size, self.n_attributes)
        elif self.mask_positioning == 'random':
            reveal_probability = torch.ones((batch_size, self.n_attributes)) / self.n_attributes
            idxs_to_reveal = torch.multinomial(reveal_probability,
                                               self.n_attributes,
                                               replacement=False)

        idxs_to_reveal = idxs_to_reveal.to(device)

        # mask indices with a redundant value
        # e.g. if idxs_to_reveal = [[3, 2, 0, 1]] and n_revealed = [[2]]
        # we define mask_idxs_to_reveal = [[1, 1, 0, 0]]
        # and we mask idxs_to_reveal with its first value, so as to get
        # idxs_to_reveal = [[3, 2, 3, 3]].
        # this way, we only have 2 indices to reveal, but we keep
        # a consistent shape within the batch.
        mask_idxs_to_reveal = torch.arange(self.n_attributes).expand(
            batch_size, self.n_attributes
        ).to(device)
        mask_idxs_to_reveal = (
            mask_idxs_to_reveal < n_revealed.view(batch_size, -1)
        ).long()
        idxs_to_reveal = idxs_to_reveal * mask_idxs_to_reveal
        idxs_to_reveal = idxs_to_reveal + idxs_to_reveal[:, 0].view(
            batch_size, 1
        ).expand(batch_size, self.n_attributes) * (1 - mask_idxs_to_reveal)

        # create attribute mask
        mask = torch.zeros((batch_size, self.n_attributes), device=sender_input.device)
        mask = mask.scatter(dim=1, index=idxs_to_reveal, value=1)
        mask = mask.detach()

        # mask attributes
        masked_input = sender_input * mask.repeat_interleave(repeats=self.n_values, dim=1)

        return masked_input, mask


    def update_curriculum_level(self):
        """
        Increments the number of revealed indices
        """
        if self.curriculum_level < self.n_attributes:
            self.curriculum_level += 1
            print('Curriculum level : ', self.curriculum_level)
