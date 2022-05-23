import os
import sys
import json

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

RELEVANT_PARAMS = [
    "random_seed",
    "n_epochs",
    "n_attributes",
    "n_values",
    "vocab_size",
    "max_len",
    "batch_size",
    "sender_cell",
    "receiver_cell",
    "train_size",
    "test_size",
    "validation_size",
    "lr",
    "sender_hidden",
    "receiver_hidden",
    "sender_emb",
    "receiver_emb",
    "acc_threshold",
    "initial_n_unmasked",
    "data_seed",
    "curriculum"
]

RELEVANT_DATA = [
    'loss',
    'acc',
    'acc_or',
    'sender_entropy',
    'length',
    'epoch'
]


def floatable(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def intable(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def parse_params_line(params_line):
    """
    The parameter line in log files looks like

    Namespace(n_attributes=10, n_values=10, receiver_cell='lstm', 
              early_stopping_thr=0.99999, fixed_length=False, ...)
    """
    params = {}
    params_list = params_line[len("Namespace("):].split(', ')
    for p in params_list:
        key, value = p.split('=', 1)
        if key in RELEVANT_PARAMS:
            if value in ['True', 'False']:
                value = (value == 'True')
            elif intable(value):
                value = int(value)
            elif floatable(value):
                value = float(value)
            else:
                value = value.strip("\"'")

            params[key] = value

    return params


def parse_results_file(file):
    """
    Reads a log file and returns a list of lines for building the 
    DataFrame object.

    Each lines contains all the relevant hyperparameters, and the 
    relevant data for a particular epoch.
    """
    df_lines = []
    for line in file.readlines():
        if line[:len('Namespace')] == 'Namespace':
            params_dict = parse_params_line(line)
            data = params_dict.copy()
        if line[0] == '{':
            line_dict = json.loads(line)
            data.update({
                key:line_dict[key]
                for key in RELEVANT_DATA if key in line_dict
            })
            df_lines.append(data)
            data = params_dict.copy()

    return df_lines


def build_dataframe(data_dir, log_file_extension=".out"):
    all_df_lines = []
    for file_name in os.listdir(data_dir):
        if file_name[-len(".out"):] == log_file_extension:
            with open(data_dir + file_name, 'r') as f:
                df_lines = parse_results_file(f)
                all_df_lines += df_lines

    df = pd.DataFrame(all_df_lines)
    df.set_index('epoch')
    return df


def plot_max_acc(df):
    """
    Finds the maximum accuracy over results of a sweep,
    and display the corresponding curve
    """
    to_plot = df.groupby([p for p in RELEVANT_PARAMS if p not in ("random_seed")])

    acc_max = 0
    name_max = ""
    for name, group in to_plot:
        new_acc = group['acc_or'].groupby('epoch').mean().max()
        if new_acc > acc_max:
            acc_max = new_acc
            name_max = name
        print(name_max, acc_max)

    sb.lineplot(x=to_plot.get_group(name_max)['epoch'], y=to_plot.get_group(name_max)['acc_or'], ci=None)
    plt.title(f"Best acc : {acc_max}")
    plt.show()


def plot_all_curves(df, param_to_plot='acc_or', title_params=None):
    """
    Plot all curves of a sweep of experiments for the 
    parameter {param_to_plot}, which needs to be in
    RELEVANT_DATA.
    Diplays the curves by groups of ten.
    """
    if param_to_plot not in RELEVANT_DATA:
        raise ValueError(f"Argument param_to_plot should be one of "
                         f"{RELEVANT_DATA}, not {param_to_plot}")

    to_plot = df.groupby([p for p in RELEVANT_PARAMS if p not in ("random_seed")])

    for i, (name, group) in enumerate(to_plot):
        if i%10 == 0:
            if i != 0: plt.show()
            fig = plt.figure(figsize=(10, 5), dpi=120)
            plt.subplots_adjust(wspace=0.5,
                                hspace=0.5)
        print(name)
        ax = fig.add_subplot(2, 5, i%10+1)
        title_indices = [RELEVANT_PARAMS.index(x)-1 for x in title_params]
        title = '\n'.join([f"{title_params[i]}: {name[title_indices[i]]}" for i in range(len(title_params))]) 
        ax.set_title(str(title).strip('{}'))
        sb.lineplot(x=group['epoch'], y=group['acc_or'], ci='sd')

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    if data_dir[-1] != '/':
        data_dir += '/'

    if len(sys.argv) > 2:
        log_file_extension = sys.argv[2]
    else:
        log_file_extension = '.out'


    df = build_dataframe(data_dir=data_dir, log_file_extension=log_file_extension)
    # plot_max_acc(df)
    plot_all_curves(df, title_params=["lr", "batch_size", "vocab_size", "max_len"])
