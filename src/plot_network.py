# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 06:00:33 2020

@author: deepu
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
from src.nn import Net
from os.path import isdir
from os import mkdir


#analyze intermediate outputs of network (conv layer 2)

#load example
checkpoint = torch.load(r"src/saved_models/model1.pkl")
network = Net()
network.load_state_dict(checkpoint['model_state_dict'])

#load sample data
Xtrain = torch.load('input/Xtrain.pt').float()
ytrain = torch.load('input/ytrain.pt')
activity_labels =pd.read_csv("input/LabelMap.csv", index_col=0)
ytrain = pd.Series(ytrain).map(activity_labels.Activity.to_dict())
subject = np.loadtxt(r'input/train/subject_train.txt', dtype = int)

metadata    = pd.DataFrame({'Subject'  : subject,
                        'Activity' : ytrain})


def create_visualization(layer_name = 'conv2', 
                         metadata   = metadata,
                         input_data = Xtrain):
    path_folder = r"src/plots/{}/".format(layer_name)
    if not isdir(path_folder):
        mkdir(path_folder)
    
    #create hook for second layer
    visualisation = {}
    def hook_fn(m, i, o):
      visualisation[m] = o 
    layer = network._modules[layer_name]
    layer.register_forward_hook(hook_fn)
      
    #run model
    out = network(input_data) 
    
    #collect visualisation
    for key, value in visualisation.items():
        print(key)
        
    #create dataframe
    records = pd.DataFrame(value.reshape([value.shape[0]*layer.out_channels,
                                          value.shape[-1]]).detach().numpy())
    
    metadata['channel'] = [[f"channel_{i}" for i in range(layer.out_channels)]]*metadata.shape[0]
    metadata            = metadata.explode('channel').reset_index(drop= True)
    records = pd.concat([metadata,records],1)
    
    #sample and reshape dataframe
    records = records.groupby(['Subject','Activity','channel']).head(1)
    records = pd.melt(records, 
                id_vars= ['Subject','Activity','channel'],
                value_vars= records.columns[3:])
    records.rename(columns = {"variable":'time'}, inplace=True) 
    
    #sample doe subjects and plot
    subset = records[records.Subject.isin(['1', '3','26', '27'])]
    g = sns.FacetGrid(subset, 
                      col = 'Activity', 
                      row = 'Subject', 
                      hue = 'channel', 
                      margin_titles=True,
                      despine = False)   
    g.map(sns.lineplot, "time", "value", dashes = True)
    g.add_legend()
    g.fig.suptitle(f'{layer_name} Layer - Subjects 1,3,26,27', y = 1.02)
    g.savefig(path_folder + 'Subjects 1,3,26,27.png')


if __name__ == '__main__':
    create_visualization(layer_name = 'conv2', 
                             metadata   = metadata,
                             input_data = Xtrain)
    create_visualization(layer_name = 'conv1', 
                             metadata   = metadata,
                             input_data = Xtrain)