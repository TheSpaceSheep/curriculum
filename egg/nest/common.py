# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json


def parse_json_sweep(config):
    config = {k: v if isinstance(v, list) else [v] for k, v in config.items()}
    perms = list(itertools.product(*config.values()))

    def to_arg(k, v):
        if type(v) in (int, float):
            return f"--{k}={v}"
        elif isinstance(v, bool):
            return f"--{k}" if v else ""
        elif isinstance(v, str):
            assert (
                '"' not in v
            ), f"Key {k} has string value {v} which contains forbidden quotes."
            return f"--{k}={v}"
        else:
            raise Exception(f"Key {k} has value {v} of unsupported type {type(v)}.")

    commands = []
    for p in perms:
        args = [to_arg(k, p[i]) for i, k in enumerate(config.keys()) if to_arg(k, p[i])]
        commands.append(args)
    return commands


def sweep(fname):
    with open(fname, "r") as config_file:
        config = json.loads(config_file.read())
    return parse_json_sweep(config)
