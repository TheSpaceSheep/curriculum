import torch
from egg.core.trainers import Trainer

# TODO: make this class abstract and inherit different curriculum trainers from it
class CurriculumTrainer(Trainer):
    def __init__(
        self,
        *args,
        curriculum_rule: str = "acc_threshold",  # fixed number of steps ? scheduler ?
        next_task_threshold=0.9,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)  

        self.curriculum_rule = curriculum_rule

        # threshold after which the learner is considered 
        # good enough to go to the next task
        self.next_task_threshold = next_task_threshold


    def update_curriculum_level(self):
        self.game.update_curriculum_level()


    def eval(self, data=None):
        """
        Wraps the eval method to schedule curriculum
        """
        mean_loss, full_interaction = super().eval(data)

        acc = full_interaction.aux["acc"].mean().item()
        acc_or = full_interaction.aux["acc_or"].mean().item()

        if acc_or > self.next_task_threshold:
            self.update_curriculum_level()

        return mean_loss, full_interaction


class PartialLoss(torch.nn.Module):
    def __init__(self, n_attributes: int, n_values: int, idx: torch.Tensor):
        """
        idx: indices to discard when computing the cross-entropy loss
        """
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.test_generalization = generalization

    def forward(
        self,
        sender_input,
        _message,
        _receiver_input,
        receiver_output,
        _labels,
        _aux_input,
    ):
        batch_size = sender_input.size(0)
        sender_input = sender_input.view(
                batch_size, self.n_attributes, self.n_values
            )[:, idx, :]
        receiver_output = receiver_output.view(
                batch_size, self.n_attributes, self.n_values
            )[:, idx, :]

        acc = (
            torch.sum(
                (
                    receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)
                ).detach(),
                dim=1,
            )
            == self.n_attributes
        ).float()
        acc_or = (
            receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)
        ).float()

        receiver_output = receiver_output.view(
            batch_size * self.n_attributes, self.n_values
        )
        labels = sender_input.argmax(dim=-1).view(batch_size * self.n_attributes)
        loss = (
            F.cross_entropy(receiver_output, labels, reduction="none")
            .view(batch_size, self.n_attributes)
            .mean(dim=-1)
        )

        return loss, {"acc": acc, "acc_or": acc_or}
