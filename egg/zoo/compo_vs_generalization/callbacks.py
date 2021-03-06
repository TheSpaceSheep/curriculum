import egg.core as core
from egg.core.interaction import Interaction
from egg.core.util import reset_optimizer_state
from egg.zoo.compo_vs_generalization.curriculum_games import CurriculumGameWrapper

import torch
import numpy as np
from typing import Optional


class CurriculumUpdater(core.Callback):
    """
    Callback that updates the curriculum according to the following strategy :
        - After each epoch, check if the current task is considered completed
        - If so, call the game's update_curriculum_level method and reset
          the optimizer's parameters

    The conditions for task completion are :
        - when acc_threshold is reached
        - if acc_threshold is None, when the accuracy becomes a plateau
          (we consider that we have a plateau when running averages over
          10 epochs are closer to each other than plateau_threshold)
    """

    def __init__(self,
                 game: CurriculumGameWrapper,
                 optimizer: torch.optim.Optimizer,
                 acc_threshold: Optional[float] = None,
                 plateau_threshold: Optional[float] = 1e-4
                 ):

        self.game = game
        self.optimizer = optimizer
        self.acc_threshold = acc_threshold
        self.plateau_threshold = plateau_threshold
        self.acc_ors = []                   # to compute running average

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

                if abs(prev_run_avg - next_run_avg) / prev_run_avg < self.plateau_threshold:
                    print("Training has converged. Updating curriculum")
                    update_curriculum = True
        elif acc_or > self.acc_threshold:
            print("Accuracy threshold reached. Updating curriculum")
            update_curriculum = True

        if update_curriculum:
            self.game.update_curriculum_level()
            reset_optimizer_state(self.game, self.optimizer)
