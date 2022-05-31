import torch
from egg.core.trainers import Trainer

class CurriculumTrainer(Trainer):
    def __init__(
        self,
        *args,
        acc_threshold: float=0.99,
        **kwargs,
    ):
        """
        Schedules the curriculum training by calling the game's
        update_curriculum_level method, each time an accuracy threshold
        is reached.  

        Params:
            acc_threshold: when accuracy reaches this threshold on the testing set,
                           the curriculum level is updated.
        
        """
        super().__init__(*args, **kwargs)  

        # threshold after which the learner is considered 
        # good enough to go to the next task
        self.acc_threshold = acc_threshold


    def eval(self, data=None):
        """
        Wraps the eval method to schedule curriculum
        """
        mean_loss, full_interaction = super().eval(data)

        acc = full_interaction.aux["acc"].mean().item()
        acc_or = full_interaction.aux["acc_or"].mean().item()

        if acc_or > self.acc_threshold:
            self.game.update_curriculum_level()
            self.game.mechanics.sender_entropy_coeff *= self.game.entropy_coeff_factor

        return mean_loss, full_interaction

