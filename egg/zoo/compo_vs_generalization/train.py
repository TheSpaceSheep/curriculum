# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
import copy
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.zoo.compo_vs_generalization.archs import (
    Freezer,
    NonLinearReceiver,
    PlusOneWrapper,
    Receiver,
    Sender,
)
from egg.zoo.compo_vs_generalization.data import (
    ScaledDataset,
    enumerate_attribute_value,
    one_hotify,
    split_train_test,
    build_datasets
)
from egg.zoo.compo_vs_generalization.intervention import Evaluator, Metrics

from egg.zoo.compo_vs_generalization.curriculum_games import GraduallyRevealAttributes
from egg.zoo.compo_vs_generalization.losses import DiffLoss, MaskedLoss, MaskedImpatientLoss
from egg.zoo.compo_vs_generalization.callbacks import CurriculumUpdater


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_attributes", type=int, default=4, help="")
    parser.add_argument("--n_values", type=int, default=4, help="")
    parser.add_argument("--data_scaler", type=int, default=100)
    parser.add_argument("--stats_freq", type=int, default=0)
    parser.add_argument(
        "--baseline", type=str, choices=["no", "mean", "builtin"], default="mean"
    )
    parser.add_argument(
        "--density_data", type=int, default=0, help="no sampling if equal 0"
    )

    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=50,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=50,
        help="Size of the hidden layer of Receiver (default: 10)",
    )

    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-2,
        help="Entropy regularisation coeff for Sender (default: 1e-2)",
    )

    parser.add_argument("--sender_cell", type=str, default="rnn")
    parser.add_argument("--receiver_cell", type=str, default="rnn")
    parser.add_argument(
        "--sender_emb",
        type=int,
        default=10,
        help="Size of the embeddings of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_emb",
        type=int,
        default=10,
        help="Size of the embeddings of Receiver (default: 10)",
    )
    parser.add_argument(
        "--early_stopping_thr",
        type=float,
        default=0.99999,
        help="Early stopping threshold on accuracy (defautl: 0.99999)",
    )
    parser.add_argument(
        "--fixed_length",
        action="store_true",
        help="Whether the messages are variable or fixed length",
    )
    parser.add_argument(
        "--build_full_dataset",
        action="store_true",
        help="Construct full dataset in memory "
             "(can fail on large input spaces!)",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default="100_000",
        help="Size of the training set",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default="1000",
        help="Size of the testing set",
    )
    parser.add_argument(
        "--validation_size",
        type=int,
        default="1000",
        help="Size of the validation set",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=714783,
        help="Random seed for data generation",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Learn according to a curriculum",
    )
    parser.add_argument(
        "--acc_threshold",
        type=float,
        default=None,
        help="Accuracy to reach before augmenting the curriculum level",
    )
    parser.add_argument(
        "--plateau_threshold",
        type=float,
        default=1e-4,
        help="Threshold to reach before considering we are on an accuracy plateau",
    )
    parser.add_argument(
        "--initial_n_unmasked",
        type=int,
        default=1,
        help="Number of unmasked attributes at the start of "
             "the curriculum training."
    )
    parser.add_argument(
        "--mask_positioning",
        type=str,
        default="random",
        help="Where the masks are located"
    )
    parser.add_argument(
        "--masking_mode",
        type=str,
        default="zero_out",
        help="How to mask the attributes"
    )
    parser.add_argument(
        "--reveal_distribution",
        type=str,
        default="deterministic",
        help="Distribution to sample the number of attributes to reveal"
    )
    parser.add_argument(
        "--impatient",
        action="store_true",
       help="make the listener impatient",
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.005,
        help="weight decay for adam optimizer"
    )
    parser.add_argument(
        "--n_revealed_test",
        type=int,
        default=10,
        help="number of attributes to reveal when testing"
    )

    args = core.init(arg_parser=parser, params=params)
    return args


def main(params):
    import copy

    opts = get_params(params)
    device = opts.device
    print(opts)

    if opts.build_full_dataset:
        print("Building full dataset of size "
              f"{opts.n_attributes ** opts.n_values}...", end="")
        full_data = enumerate_attribute_value(opts.n_attributes, opts.n_values)

        train, test_data = split_train_test(full_data, 0.2)
        test, validation = split_train_test(test_data, 0.5)

    else:
        rng = torch.Generator()
        rng.manual_seed(opts.data_seed)
        print("Building train, test and validation sets "
              f"of size {opts.train_size}, {opts.test_size}, "
              f"and {opts.validation_size}...", end="")
        train, test, validation = \
            build_datasets(opts.n_attributes,
                           opts.n_values,
                           opts.train_size,
                           opts.test_size,
                           opts.validation_size,
                           rng=rng)

    if opts.masking_mode == 'dedicated_value':
        # Add one value that represents masking
        # the corresponding attribute
        tot_n_values = opts.n_values + 1
    else:
        tot_n_values = opts.n_values

    train, validation, test = [
        one_hotify(x, opts.n_attributes, tot_n_values)
        for x in [train, validation, test]
    ]

    train = ScaledDataset(train, opts.data_scaler)
    validation = ScaledDataset(validation, 1)
    test = ScaledDataset(test, 1)
    print(" - done")

    test_loader = DataLoader(test, batch_size=opts.batch_size)
    train_loader = DataLoader(train, batch_size=opts.batch_size)
    validation_loader = DataLoader(validation, batch_size=len(validation))

    n_dim = opts.n_attributes * tot_n_values

    if opts.receiver_cell not in ["lstm", "rnn", "gru"]:
        raise ValueError(f"Unknown receiver cell, {opts.receiver_cell}")


    receiver = Receiver(n_hidden=opts.receiver_hidden, n_outputs=n_dim)
    if opts.impatient:
        receiver = core.RnnReceiverImpatient(
            receiver,
            opts.vocab_size + 1,
            opts.receiver_emb,
            opts.receiver_hidden,
            cell=opts.receiver_cell,
        )
    else:
        receiver = core.RnnReceiverDeterministic(
            receiver,
            opts.vocab_size + 1,
            opts.receiver_emb,
            opts.receiver_hidden,
            cell=opts.receiver_cell,
        )

    if opts.sender_cell in ["lstm", "rnn", "gru"]:
        sender = Sender(n_inputs=n_dim, n_hidden=opts.sender_hidden)
        sender = core.RnnSenderReinforce(
            agent=sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_emb,
            hidden_size=opts.sender_hidden,
            max_len=opts.max_len,
            cell=opts.sender_cell,
        )
    else:
        raise ValueError(f"Unknown sender cell, {opts.sender_cell}")

    if opts.fixed_length:
        sender = PlusOneWrapper(sender)  # to enforce fixed-length messages

    baseline = {
        "no": core.baselines.NoBaseline,
        "mean": core.baselines.MeanBaseline,
        "builtin": core.baselines.BuiltInBaseline,
    }[opts.baseline]

    if opts.impatient:
        loss = MaskedImpatientLoss(opts.n_attributes, tot_n_values)
    elif opts.curriculum and opts.masking_mode != "dedicated_value":
        loss = MaskedLoss(opts.n_attributes, tot_n_values)
    else:
        loss = DiffLoss(opts.n_attributes, tot_n_values)

    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=0.0,
        length_cost=0.0,
        baseline_type=baseline,
    )

    if opts.curriculum:
        # wrap game
        game = GraduallyRevealAttributes(
            game,
            opts.n_attributes,
            tot_n_values,
            mask_positioning=opts.mask_positioning,
            masking_mode=opts.masking_mode,
            reveal_distribution=opts.reveal_distribution,
            initial_n_unmasked=opts.initial_n_unmasked,
            n_revealed_test=opts.n_revealed_test
        )

    optimizer = torch.optim.Adam(game.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

    metrics_evaluator = Metrics(
        validation.examples,
        opts.device,
        opts.n_attributes,
        tot_n_values,
        opts.vocab_size + 1,
        freq=opts.stats_freq,
    )

    loaders = []
    loaders.append(
        (
            "test set",
            test_loader,
            DiffLoss(opts.n_attributes, tot_n_values),
        )
    )
    loaders.append(
        (
            "validation set",
            validation_loader,
            DiffLoss(opts.n_attributes, tot_n_values),
        )
    )

    holdout_evaluator = Evaluator(loaders, opts.device, freq=0)
    interaction_saver = core.InteractionSaver(
        test_epochs=list(range(1, opts.n_epochs, opts.stats_freq)),
        checkpoint_dir=f"{sys.argv[1]}")

    callbacks = [
        core.ConsoleLogger(as_json=True, print_train_loss=True),
        metrics_evaluator,
        holdout_evaluator,
        interaction_saver,
    ]

    if opts.curriculum:
        curriculum_manager = CurriculumUpdater(
            game,
            optimizer,
            acc_threshold=opts.acc_threshold,
            plateau_threshold=opts.plateau_threshold)
        callbacks.append(curriculum_manager)

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=callbacks
    )

    print("Beginning training")

    trainer.train(n_epochs=opts.n_epochs)

    print("---End--")

    core.close()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
