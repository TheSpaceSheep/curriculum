# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import random
from egg.zoo.compo_vs_generalization.bst import BinarySearchTree

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


def one_hotify(data, n_attributes, n_values):
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


def build_test_set(n_attributes, n_values, size):
    """
    Samples {size} elements from the dataset, with no possible repeat
    """
    max_size = 10e6  # TODO: find better value for this empirically
    if size > max_size:
        print(f"Warning : trying to build a test set of size greater than {max_size} which might take a long time")

    # keep data in a tree for faster lookup
    data_tree = BinarySearchTree()

    data = []
    i = 0
    while i < size:
        sample = tuple([random.randint(0, n_values-1) for _ in range(n_attributes)])
        if sample not in data_tree:
           data_tree.insert(sample)
           data.append(sample)
           i += 1
        else:
            print('sample already in validation set')

    assert len(data) == len(set(data)), "Found duplicates in test set"

    return data


def build_datasets(n_attributes, n_values, train_size, test_size, validation_size):
    """
    Returns the train, test and validation sets by sampling the data.
    Samples can be repeated in the training set but not in the testing set
    Also there is no overlap between train and test set.
    """
    # prevent infinite loops
    assert test_size < train_size and validation_size < train_size, 
      "please set the --train_size option to be (much) larger than test and validation sizes."

    # create test and validation sets simultaneously to prevent repeats
    all_test_data = build_test_set(n_attributes, n_values, size=test_size+validation_size)
    test_set = all_test_data[:test_size]
    validation_set = all_test_data[test_size:]

    # build tree to speed up the search
    test_set_tree = BinarySearchTree()
    for sample in test_set:
        test_set_tree.insert(sample)

    train_set = []
    i = 0
    while i < train_size:
        sample = tuple([random.randint(0, n_values-1) for _ in range(n_attributes)])
        if sample not in test_set_tree:
           train_set.append(sample)
           i += 1

    for s in train_set:
        assert s not in test_set, "Found overlap between train and test set"

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


if __name__ == "__main__":
    dataset = enumerate_attribute_value(n_attributes=2, n_values=10)
    train, holdout = split_holdout(dataset)
    print(len(train), len(holdout), len(dataset))

    print([x[0] for x in [train, holdout, dataset]])
