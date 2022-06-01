import egg.core as core
from egg.core.interaction import Interaction
from egg.core.util import reset_optimizer_state
from egg.zoo.compo_vs_generalization.curriculum_games import CurriculumGameWrapper

import torch
import numpy as np


class CurriculumManager(core.Callback):
    def __init__(self, 
            game: CurriculumGameWrapper, 
            optimizer: torch.optim.Optimizer,
            acc_threshold=None):

        self.game = game
        self.optimizer = optimizer
        self.acc_threshold = acc_threshold
        self.acc_ors = []

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        acc_or = logs.aux["acc_or"].mean().item()
        update_curriculum = False
        if self.acc_threshold is None:
            self.acc_ors.append(acc_or)
            if len(self.acc_ors) > 20:
                del self.acc_ors[0]

                # compute running averages
                prev_run_avg = np.array(self.acc_ors[:10]).mean()
                next_run_avg = np.array(self.acc_ors[10:]).mean()
                print(abs(prev_run_avg - next_run_avg)/prev_run_avg)

                if abs(prev_run_avg - next_run_avg)/prev_run_avg < 1e-4:
                    print("Training has converged. Updating curriculum")
                    update_curriculum = True
        elif acc_or > self.acc_threshold:
            print("Accuracy threshold reached. Updating curriculum")
            update_curriculum = True
        
        if update_curriculum:
            self.game.update_curriculum_level()
            reset_optimizer_state(self.game, self.optimizer)




