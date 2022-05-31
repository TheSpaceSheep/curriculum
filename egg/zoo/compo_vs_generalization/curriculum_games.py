import torch
import torch.nn as nn
from abc import abstractmethod
from egg.core.reinforce_wrappers import SenderReceiverRnnReinforce
from egg.zoo.compo_vs_generalization.data import mask_attributes
from egg.zoo.compo_vs_generalization.losses import MaskedLoss

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
    In this game wrapper, all attributes are masked except {n_unmasked} of them.
    The position of unmasked attributes can be either random if {mode} is 
    'random', or at the left if {mode} is 'from_left_to_right'.
    During training, more attributes are gradually revealed as n_unmasked
    augments.
    """
    def __init__(self,
            game: SenderReceiverRnnReinforce,
            n_attributes: int,
            n_values: int,
            mask_positioning: str,
            masking_mode: str,
            initial_n_unmasked: int=1
        ):
        valid_mask_positionings = ['left_to_right', 'random']
        if mask_positioning not in valid_mask_positionings:
            raise ValueError(f"Invalid mask_positioning {mask_positioning}. mode should be in {valid_mask_positionings}")

        valid_masking_modes = ['zero_out', 'dedicated_value']
        if masking_mode not in valid_masking_modes:
            raise ValueError(f"Invalid masking_mode {masking_mode}. mode should be in {valid_masking_modes}")

        super().__init__(game)
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.mask_positioning = mask_positioning
        self.masking_mode = masking_mode
        self.n_unmasked = min(initial_n_unmasked, n_attributes)


    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        batch_size = sender_input.shape[0]

        if self.mask_positioning == 'left_to_right':
            idxs_to_mask = torch.arange(self.n_unmasked, self.n_attributes, dtype=torch.long)
            idxs_to_mask = idxs_to_mask.expand(batch_size, idxs_to_mask.shape[0])
        elif self.mask_positioning == 'random':
            mask_probability = torch.ones((batch_size, self.n_attributes))/self.n_attributes
            n_masks = self.n_attributes - self.n_unmasked
            # multinomial throws an error when we try to sample 0 elements
            # so we manually specify an empty idxs batch
            if n_masks == 0:
                idxs_to_mask = torch.tensor([[]]*batch_size)
            else:
                idxs_to_mask = torch.multinomial(mask_probability,
                        n_masks,
                        replacement=False)

        idxs_to_mask = idxs_to_mask.to(sender_input.device)
        sender_input = mask_attributes(sender_input,
                idxs_to_mask,
                self.n_attributes,
                self.n_values,
                mask_by_last_value=(self.masking_mode=='dedicated_value'))

        # pass indices to mask to the loss function through aux_input
        if aux_input is None:
            aux_input = {}
        aux_input['idxs_to_mask'] = idxs_to_mask


        return self.game(sender_input, labels, receiver_input=None, aux_input=aux_input)


    def update_curriculum_level(self):
        """
        Increments the number of revealed indices
        """
        if self.n_unmasked < self.n_attributes:
            self.n_unmasked += 1
            print('Curriculum level : ', self.n_unmasked)

