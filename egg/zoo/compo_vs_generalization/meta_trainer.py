import os
import pathlib
from typing import List, Optional
import egg.core as core
import higher

import torch

try:
    # requires python >= 3.7
    from contextlib import nullcontext
except ImportError:
    # not exactly the same, but will do for our purposes
    from contextlib import suppress as nullcontext

from egg.core.batch import Batch
from egg.core.interaction import Interaction


class MetaTrainer(core.Trainer):
    """
    Trainer that implements meta-training on tasks specified in meta_tasks
    
    :param meta_tasks: List of ints that represent task complexity (can be assigned to
    the curriculum_level attribute of the game)

    
    """

    def __init__(
        self,
        *args,
        meta_tasks: List[int] = None,
        meta_lr: float = 0.001,
        **kwargs
    ):
        """
        :param meta_tasks: A list of curriculum levels for the agents to meta-train on.
        args and kwargs should be the arguments passed to a normal Trainer instance.
        """
        
        super().__init__(*args, **kwargs)
        self.meta_tasks = meta_tasks
        self.meta_optimizer = torch.optim.Adam(self.game.parameters(), lr=meta_lr)


    def train_epoch(self):
        mean_loss = 0
        n_batches = 0
        interactions = []

        n_inner_iter = 3  # only one meta learning step

        self.game.train()

        self.optimizer.zero_grad()

        for batch_id, batch in enumerate(self.train_data):
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to(self.device)

            context = autocast() if self.scaler else nullcontext()
            old_curri_level = self.game.curriculum_level
            with context, torch.backends.cudnn.flags(enabled=False):

                self.meta_optimizer.zero_grad()
                # meta_training
                for level in self.meta_tasks:
                    # set the game task
                    self.game.curriculum_level = level
                    with higher.innerloop_ctx(
                        self.game,
                        self.optimizer,
                        copy_initial_weights=True
                    ) as (fgame, diffopt):

                        for _ in range(n_inner_iter):
                            optimized_loss, _ = fgame(*batch)

                            if self.update_freq > 1:
                                # throughout EGG, we minimize _mean_ loss, not sum
                                # hence, we need to account for that when aggregating grads
                                optimized_loss = optimized_loss / self.update_freq

                            if batch_id % self.update_freq == self.update_freq - 1:
                                if self.grad_norm:
                                    torch.nn.utils.clip_grad_norm_(
                                        fgame.parameters(), self.grad_norm
                                    )
                                diffopt.step(optimized_loss)

                        # meta optimization
                        meta_loss, interaction = fgame(*batch)
                        meta_loss.backward()  # aggregate meta-gradient for task i

                        n_batches += 1
                        mean_loss += meta_loss.detach()
                        if (
                            self.distributed_context.is_distributed
                            and self.aggregate_interaction_logs
                        ):
                            interaction = Interaction.gather_distributed_interactions(interaction)
                        interaction = interaction.to("cpu")

                        for callback in self.callbacks:
                            callback.on_batch_end(interaction, meta_loss, batch_id)

                        interactions.append(interaction)



                # optimization of models meta-parameters
                self.meta_optimizer.step()



        #if self.optimizer_scheduler:
        #    self.optimizer_scheduler.step()

        self.game.curriculum_level = old_curri_level
        mean_loss /= (n_batches*len(self.meta_tasks))
        full_interaction = Interaction.from_iterable(interactions)
        return mean_loss.item(), full_interaction

