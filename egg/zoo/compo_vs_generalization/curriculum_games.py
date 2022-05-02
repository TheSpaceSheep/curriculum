import torch
from egg.core.reinforce_wrappers import SenderReceiverRnnReinforce
from egg.zoo.compo_vs_generalization.data import mask_attributes
from egg.zoo.compo_vs_generalization.losses import MaskedLoss


class GraduallyRevealAttributes(SenderReceiverRnnReinforce):
    """
    In this game, all attributes are masked except the first {curriculum level} ones.
    At first, only the first (leftmost) attribute is visible, then at each curriculum update,
    the next attribute is revealed, until all attributes are visible.

    """
    def __init__(self,
            n_attributes: int,
            n_values: int,
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.curriculum_level = 1


    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        batch_size = sender_input.shape[0]
        idx = torch.tensor(range(self.curriculum_level, self.n_attributes), dtype=torch.long)
        idx = idx.expand(batch_size, idx.shape[0])

        # mask all attributes except the first {curriculum_level} ones.
        sender_input = mask_attributes(sender_input, idx, self.n_attributes, self.n_values)

        # adapt loss function to the current mask
        self.loss.mask_idx = idx


        return super().forward(sender_input, labels, receiver_input=None, aux_input=None)
    

    def update_curriculum_level(self):
        if self.curriculum_level < self.n_attributes:
            self.curriculum_level += 1
            print('Curriculum level : ', self.curriculum_level)

