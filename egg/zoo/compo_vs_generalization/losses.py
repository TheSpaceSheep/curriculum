import torch
import torch.nn.functional as F
from egg.zoo.compo_vs_generalization.data import mask_attributes


class DiffLoss(torch.nn.Module):
    def __init__(self, n_attributes, n_values, generalization=False):
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
        sender_input = sender_input.view(batch_size, self.n_attributes, self.n_values)
        receiver_output = receiver_output.view(
            batch_size, self.n_attributes, self.n_values
        )

        if self.test_generalization:
            acc, acc_or, loss = 0.0, 0.0, 0.0

            for attr in range(self.n_attributes):
                zero_index = torch.nonzero(sender_input[:, attr, 0]).squeeze()
                masked_size = zero_index.size(0)
                masked_input = torch.index_select(sender_input, 0, zero_index)
                masked_output = torch.index_select(receiver_output, 0, zero_index)

                no_attribute_input = torch.cat(
                    [masked_input[:, :attr, :], masked_input[:, attr + 1 :, :]], dim=1
                )
                no_attribute_output = torch.cat(
                    [masked_output[:, :attr, :], masked_output[:, attr + 1 :, :]], dim=1
                )

                n_attributes = self.n_attributes - 1
                attr_acc = (
                    (
                        (
                            no_attribute_output.argmax(dim=-1)
                            == no_attribute_input.argmax(dim=-1)
                        ).sum(dim=1)
                        == n_attributes
                    )
                    .float()
                    .mean()
                )
                acc += attr_acc

                attr_acc_or = (
                    (
                        no_attribute_output.argmax(dim=-1)
                        == no_attribute_input.argmax(dim=-1)
                    )
                    .float()
                    .mean()
                )
                acc_or += attr_acc_or
                labels = no_attribute_input.argmax(dim=-1).view(
                    masked_size * n_attributes
                )
                predictions = no_attribute_output.view(
                    masked_size * n_attributes, self.n_values
                )
                # NB: THIS LOSS IS NOT SUITABLY SHAPED TO BE USED IN REINFORCE TRAINING!
                loss += F.cross_entropy(predictions, labels, reduction="mean")

            acc /= self.n_attributes
            acc_or /= self.n_attributes
        else:
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


class MaskedLoss(torch.nn.Module):
    def __init__(
        self,
        n_attributes: int,
        n_values: int
    ):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

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
        mask = _aux_input['mask']

        sender_input = sender_input.view(
            batch_size, self.n_attributes, self.n_values
        )
        receiver_output = receiver_output.view(
            batch_size, self.n_attributes, self.n_values
        )

        # matches for each attributes
        matches = (sender_input.argmax(dim=-1) == receiver_output.argmax(dim=-1)).float()

        # Average across attributes (but only count revealed attributes)
        acc_or = (matches * mask).sum(-1) / mask.sum(-1)

        # Exact matches: Count 1 if attribute is predicted correctly OR if it is masked.
        acc = torch.all((matches == 1) | (mask == 0), dim=-1).float()

        # Loss for each attribute (you need to flatten the first dimensions
        # and reshape them afterwards)
        loss_by_attributes = F.cross_entropy(
            receiver_output.view(-1, self.n_attributes),
            sender_input.argmax(-1).view(-1),
            reduction="none",
        ).view(batch_size, self.n_attributes)

        # Take the mean, but only of the revealed attributes
        loss = (loss_by_attributes * mask).sum(-1) / mask.sum(-1)

<<<<<<< HEAD
        return loss, {"acc": acc, "acc_or": acc_or}


class MaskedImpatientLoss(torch.nn.Module):
    def __init__(
        self,
        n_attributes: int,
        n_values: int
    ):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

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
        seq_len = receiver_output.shape[1]
        idxs_to_reveal = _aux_input['idxs_to_reveal']

        sender_input = sender_input.reshape(
            batch_size, self.n_attributes, self.n_values
        )
        receiver_output = receiver_output.reshape(
            seq_len, batch_size, self.n_attributes, self.n_values
        )

        # create mask for symbols after EOS token
        len_mask_idx = torch.arange(seq_len, dtype=torch.long).to(sender_input.device)
        len_mask_idx = len_mask_idx.expand(batch_size, seq_len)
        len_mask = (len_mask_idx <= _message.argmin(1).reshape(batch_size, 1)).float().t()
        len_mask = len_mask.unsqueeze(-1).expand(seq_len, batch_size, self.n_attributes)

        # create attribute mask
        mask = torch.zeros_like(idxs_to_reveal)
        mask = mask.scatter(
            dim=1, index=idxs_to_reveal, value=1.
        ).float()
        mask = mask.expand(seq_len, batch_size, self.n_attributes)

        # matches for each attributes at each timestep
        matches = (
            sender_input.argmax(
                dim=- 1).expand(
                seq_len,
                batch_size,
                self.n_attributes) == receiver_output.argmax(
                dim=- 1)).float()

        # Average across attributes (but only count revealed attributes)
        positional_acc_or = (matches * len_mask * mask).sum(dim=-1) / mask.sum(dim=-1).mean(0)
        acc_or = positional_acc_or.sum(dim=0) / len_mask.sum(dim=0).mean(-1)

        # Exact matches: Count 1 if attribute is predicted correctly OR if it is masked.
        acc = torch.all((matches == 1) | (mask * len_mask == 0), dim=-1).float().sum(dim=0) / len_mask.sum(dim=0).mean(-1)

        # Loss for each attribute (you need to flatten the first dimensions
        # and reshape them afterwards)
        loss_by_attributes = F.cross_entropy(
            receiver_output.reshape(-1, self.n_values),
            sender_input.expand(seq_len, batch_size, self.n_attributes, self.n_values).argmax(-1).reshape(-1),
            reduction="none",
        ).reshape(seq_len, batch_size, self.n_attributes)

        # Take the mean, but only of the revealed attributes

        loss = (loss_by_attributes * mask * len_mask).sum(dim=(0, -1))

        loss = loss / mask.sum(dim=-1).mean(0)
        # loss = loss / len_mask.sum(dim=0).mean(-1)

        return loss, {"acc": acc, "acc_or": acc_or}
=======
        logs = {
            "acc": acc,
            "acc_or": acc_or,
            "curriculum_level": _aux_input["curriculum_level"]
        }

        return loss, logs

>>>>>>> 053aed9... pass curriculum level through eggs logging system
