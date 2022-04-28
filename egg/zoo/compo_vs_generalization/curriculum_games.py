import torch
from egg.core.reinforce_wrappers import SenderReceiverRnnReinforce
from egg.zoo.compo_vs_generalization.data import mask_attributes


class MaskingGame(SenderReceiverRnnReinforce):
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
        # mask all attributes except the first {curriculum_level} ones.
        idx = torch.tensor(range(self.curriculum_level, self.n_attributes), dtype=torch.long)
        sender_input = mask_attributes(sender_input, idx, self.n_attributes, self.n_values)

        return super().forward(sender_input, labels, receiver_input=None, aux_input=None)

    def update_curriculum_level(self):
        if self.curriculum_level < self.n_attributes:
            self.curriculum_level += 1
            print('Curriculum level : ', self.curriculum_level)


