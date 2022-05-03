import torch
from abc import abstracmethod
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
            mode: str
            initial_n_unmasked: int=1
        ):
        valid_modes = ['left_to_right', 'random']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode {mode}. mode should be in {valid_modes}")
        super().__init__(game)
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.mode = mode
        self.n_unmasked = initial_n_unmasked


    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        batch_size = sender_input.shape[0]

        if self.mode = 'left_to_right':
            idxs_to_mask = torch.arange(self.n_unmasked, self.n_attributes), dtype=torch.long).to(sender_input.device)
            idxs_to_mask = idxs_to_mask.expand(batch_size, idxs_to_mask.shape[0])
        elif self.mode = 'random':
            mask_probability = torch.ones((batch_size, self.n_attributes))/self.n_attributes
            idxs_to_mask = torch.multinomial(mask_probability,
                    self.n_attributes - self.n_unmasked,
                    replacement=False)

        sender_input = mask_attributes(sender_input, idxs_to_mask, self.n_attributes, self.n_values)

        # pass indices to mask to the loss function through aux_input
        if aux_input is None:
            aux_input = {}
        aux_input['idxs_to_mask'] = idxs_to_mask


        return self.game(sender_input, labels, receiver_input=None, aux_input=None)
    

    def update_curriculum_level(self):
        """
        Increments the number of revealed indices
        """
        if self.n_unmasked < self.n_attributes:
            self.n_unmasked += 1
            print('Curriculum level : ', self.n_unmasked)

