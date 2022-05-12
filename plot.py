import os
import sys
import json
import matplotlib.pyplot as plt

# Simple script for plotting results from all log files in a specified folder
# Example : python3 plot.py data_dir/ .out

data_dir = sys.argv[1]
if data_dir[-1] != '/':
    data_dir += '/'

if len(sys.argv) > 2:
    log_file_extension = sys.argv[2]
else:
    log_file_extension = '.out'



def parse_params_line(params_line):
    # TODO: extract relevant parameters
    # to display on the plot
    return params_line

def plot_file(file):
    results = {
        'loss': [],
        'acc': [],
        'acc_or': [],
        'sender_entropy': [],
        'length': []
    }
            
    for line in file.readlines():
        params_line = ""
        if line[:9] == 'Namespace':
            params_line = line
            print(params_line)
        if line[0] == '{':
            line_dict = json.loads(line)
            for key in line_dict:
                if key in results and 'entropy' not in key and 'length' not in key:
                    results[key].append(line_dict[key])
    for key in results:
        if 'entropy' not in key and 'length' not in key:
            plt.plot(results[key])
            plt.xlabel("epoch")
            plt.ylabel(key)
            plt.title(parse_params_line(params_line))
            plt.show()



for file_name in os.listdir(data_dir):
    if file_name[-4:] == log_file_extension:
        with open(data_dir + file_name, 'r') as file:
            plot_file(file)
