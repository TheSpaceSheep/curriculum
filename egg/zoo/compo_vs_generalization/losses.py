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
    def __init__(self, 
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
        idxs_to_reveal = _aux_input['idxs_to_reveal']
        n_revealed = _aux_input['n_revealed']

        masked_input = mask_attributes(sender_input,
                idxs_to_reveal,
                self.n_attributes,
                self.n_values)
        masked_output = mask_attributes(receiver_output,
                idxs_to_reveal,
                self.n_attributes,
                self.n_values)

        masked_input = masked_input.view(
                batch_size, -1, self.n_values
            )
        masked_output = masked_output.view(
                batch_size, -1, self.n_values
            )

        # acc : (batch_size,) tensor containing
        # 1 where the vectors were perfectly
        # reconstructed by the receiver, 0 elsewhere.
        acc = (
            (
                torch.sum(
                    (
                        # this counts all masks as correct predictions
                        masked_output.argmax(dim=-1) == masked_input.argmax(dim=-1)
                    ).detach(),
                    dim=1,
                ) 
                # so we have to compensate by 
                # substracting the number of masks
                - (self.n_attributes - n_revealed)
            )
            == n_revealed
        ).float()

        masked_output = masked_output.view(
                -1, self.n_values
            )
        masked_input = masked_input.view(
                -1, self.n_values
            )
        # remove masked values before computing acc_or and loss !
        masked_output = masked_output[torch.abs(masked_output).sum(dim=1) != 0]
        masked_input = masked_input[torch.abs(masked_input).sum(dim=1) != 0]

        # acc_or : (batch_size * < n_attributes) tensor
        # containing 1 where a revealed attribute was
        # correctly reconstructed by the receiver,
        # (even if the vector as a whole is not
        # perfectly reconstructed), 0 elsewhere.
        acc_or = (
                masked_output.argmax(dim=-1) == masked_input.argmax(dim=-1)
        ).float()
        # taking the mean of acc (resp. acc_or) gives the average accuracy
        # of vector reconstruction (resp. attribute reconstruction)

        # these labels are incorrect !
        labels = masked_input.argmax(dim=-1)
        print(masked_output)
        print(masked_input)
        print(labels)
        # labels = labels.scatter(dim=1, index=idxs_to_mask, value=-1)
        # print(labels.shape)
        # labels = labels[labels != -1]
        # print(labels.shape)
        # labels = labels.view(-1)
        # print(labels.shape)

        # sender_input = sender_input.view(
        #         -1, self.n_values
        # )
        # receiver_output = receiver_output.view(
        #         -1, self.n_values
        # )

        # labels = sender_input.argmax(dim=-1)
        # labels = labels.view(-1)

        loss = (
            F.cross_entropy(masked_output, labels, reduction="none")
            .mean(dim=-1)
        )

        return loss, {"acc": acc, "acc_or": acc_or}

