# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import random
from typing import Tuple, List

import torch


def enumerate_attribute_value(n_attributes, n_values):
    """
    Returns the ENTIRE dataset (not suitable for very large input spaces)
    """
    iters = [range(n_values) for _ in range(n_attributes)]

    return list(itertools.product(*iters))


def select_subset_V1(data, n_subset, n_attributes, n_values, random_seed=7):
    import numpy as np

    assert n_subset <= n_values
    random_state = np.random.RandomState(seed=random_seed)

    chosen_val = []
    for attribute in range(n_attributes):
        chosen_val.append(
            [0]
            + list(random_state.choice(range(1, n_values), n_subset - 1, replace=False))
        )

    sampled_data = []
    for sample in data:
        boolean = True
        for attribute in range(n_attributes):
            boolean = boolean and (sample[attribute] in chosen_val[attribute])
        if boolean:
            sampled_data.append(sample)
    return sampled_data


def select_subset_V2(data, n_subset, n_attributes, n_values, random_seed=7):
    import numpy as np

    assert n_subset <= n_values
    random_state = np.random.RandomState(seed=random_seed)
    sampled_data = []
    # Sample the diagonal (minus (0,0)) to impose having each attribute is present at least once in the dataset
    start = 0
    while start < (n_values ** n_attributes):
        if start > 0:
            sampled_data.append(data[start])
        start += n_values + 1
    # Sample remaining
    to_sample = (n_subset ** n_attributes) - len(sampled_data)
    tobesampled = copy.deepcopy(data)
    for sample in sampled_data:
        tobesampled.remove(sample)
    tmp = list(random_state.choice(range(len(tobesampled)), to_sample, replace=False))

    for i in tmp:
        sampled_data += [tobesampled[i]]
    return sampled_data


def one_hotify(data: List[Tuple[int]], n_attributes: int, n_values: int) -> List[torch.Tensor]:
    """
    Params:
        data: list of tuples of length n_attributes
              [(a1, ..., an), (a2, ..., an), ...]
        n_attributes: int
        n_values: int

    Returns :
        r: list of flattened one-hot matrices
           [r1, r2, ...] where the ri are 1D torch
           tensors of shape (n_attributes x n_values)
    """
    r = []
    for config in data:
        z = torch.zeros((n_attributes, n_values))
        for i in range(n_attributes):
            z[i, config[i]] = 1
        r.append(z.view(-1))
    return r


def split_holdout(dataset):
    train, hold_out = [], []

    for values in dataset:
        indicators = [x == 0 for x in values]
        if not any(indicators):
            train.append(values)
        elif sum(indicators) == 1:
            hold_out.append(values)
        else:
            pass

    return train, hold_out


def split_train_test(dataset, p_hold_out=0.1, random_seed=7):
    import numpy as np

    assert p_hold_out > 0
    random_state = np.random.RandomState(seed=random_seed)

    n = len(dataset)
    permutation = random_state.permutation(n)

    n_test = int(p_hold_out * n)

    test = [dataset[i] for i in permutation[:n_test]]
    train = [dataset[i] for i in permutation[n_test:]]
    assert train and test

    assert len(train) + len(test) == len(dataset)

    return train, test


def build_random_dataset(n_attributes: int,
                         n_values: int,
                         size: int,
                         data_to_exclude: set = None,
                         allow_duplicates: bool = False,
                         rng: torch.Generator = None
                         ) -> List[Tuple[int]]:
    """
    Samples {size} elements from the input space,
    with no overlap with the set {data_to_exclude},
    """
    if n_values**n_attributes / size < 10e4 and allow_duplicates:
        print(f"Warning : Building a dataset that probably contains duplicates")

    if data_to_exclude is None:
        data_to_exclude = set()

    # use a list if duplicates are allowed (uses less memory than set)
    data = list() if allow_duplicates else set()

    while len(data) < size:
        sample = tuple(torch.randint(n_values, size=(n_attributes,), generator=rng).tolist())
        if sample not in data_to_exclude:
            if allow_duplicates:
                data.append(sample)
            else:
                data.add(sample)

    return list(data)


def build_datasets(n_attributes: int,
                   n_values: int,
                   train_size: int,
                   test_size: int,
                   validation_size: int,
                   rng: torch.Generator = None) -> Tuple[List[Tuple[int]], List[Tuple[int]], List[Tuple[int]]]:
    """
    Returns the train, test and validation sets by sampling the data.
    Samples can be repeated in the training set but not in the testing set
    Also there is no overlap between train and test set.
    """
    # create test and validation sets simultaneously to prevent repeats
    all_test_data = build_random_dataset(n_attributes,
                                         n_values,
                                         size=test_size + validation_size,
                                         allow_duplicates=False,
                                         rng=rng)
    test_set = all_test_data[:test_size]
    validation_set = all_test_data[test_size:]

    train_set = build_random_dataset(n_attributes,
                                     n_values,
                                     size=train_size,
                                     data_to_exclude=all_test_data,
                                     allow_duplicates=False,  # it might be necessary to change this for very large datasets
                                     rng=rng
                                     )

    return train_set, test_set, validation_set


class ScaledDataset:
    def __init__(self, examples, scaling_factor=1):
        self.examples = examples
        self.scaling_factor = scaling_factor

    def __len__(self):
        return len(self.examples) * self.scaling_factor

    def __getitem__(self, k):
        k = k % len(self.examples)
        return self.examples[k], torch.zeros(1)


def mask_attributes(sender_input,
                    idxs_to_reveal,
                    n_attributes,
                    n_values,
                    mask_by_last_value=False):
    """
    sender_input: data to mask (already one-hotified)| (batch_size, n_attributes*n_values)
    idxs_to_reveal: indices of attributes to reveal  | (batch_size, < n_attributes)
    """
    assert sender_input.shape[0] == idxs_to_reveal.shape[0],  \
        f"Cannot mask, batch_sizes do not match between input ({sender_input.shape[0]}) and indices ({idxs_to_reveal.shape[0]})"

    batch_size = sender_input.shape[0]

    masked_input = sender_input * mask
    if mask_by_last_value:
        # infer indices of masked attributes from masked_input
        idxs_to_mask = (torch.abs(masked_input.view(
            batch_size * n_attributes, n_values
        )).sum(dim=1) == 0).nonzero()

        # indices of last values in the one hot vector,
        # that should be set to one
        idxs_to_one = (idxs_to_mask + 1) * n_values - 1

        add_mask = torch.zeros(
            (batch_size * n_attributes * n_values),
            device=sender_input.device
        )
        add_mask = add_mask.scatter(
            dim=0, index=idxs_to_one, value=1
        )
        add_mask = add_mask.view(
            batch_size, n_attributes * n_values
        )
        masked_input += add_mask

    return masked_input


if __name__ == "__main__":
    dataset = enumerate_attribute_value(n_attributes=2, n_values=10)
    train, holdout = split_holdout(dataset)
    print(len(train), len(holdout), len(dataset))

    print([x[0] for x in [train, holdout, dataset]])
