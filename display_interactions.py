import torch
import os
import sys
from plot_dataframe import build_dataframe

data_dir = sys.argv[1]
if data_dir[-1] != '/':
    data_dir += '/'

interactions_dir = data_dir + "interactions/validation/"

 

def display_interaction(interaction):
    input_batch = interaction.sender_input
    message_batch = interaction.message
    
    df = build_dataframe(data_dir)

    for i, m in zip(input_batch, message_batch):
        n_attributes, n_values = df['n_attributes'].iloc[0], df['n_values'].iloc[0]
        reshaped_i = i.view(n_attributes, n_values).argmax(dim=-1)
        print(' '.join([str(x) for x in reshaped_i.tolist()]))
        print(' '.join([str(x) for x in m.tolist()]))


for epoch_dir in [d.name for d in os.scandir(interactions_dir) if d.is_dir()][-10:]:
    print(epoch_dir)
    for filename in os.listdir(interactions_dir + epoch_dir):
        interaction = torch.load(interactions_dir+epoch_dir+'/'+filename)
        display_interaction(interaction)


