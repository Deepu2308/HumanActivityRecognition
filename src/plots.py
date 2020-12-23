# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:45:20 2020

@author: deepu
"""

import torch
import pandas as pd
import os
import numpy as np
import seaborn as sns

#labels
ytrain = torch.load('input/ytrain.pt')
ytest  = torch.load('input/ytest.pt')

#label mapper
labels = pd.read_csv('input/activity_labels.txt', 
                     header=None, 
                     sep = ' ',
                     index_col= 0
).iloc[:,0].to_dict()
train_labels = list(map(lambda x: labels[x.item()+1], ytrain))



#inertial data
train_filename    = [ i for i in \
                         os.listdir(r'input/train/Inertial Signals/')]

    
data_dict, subject = {},np.loadtxt(r'input/train/subject_train.txt')
for i in train_filename:
    data = pd.DataFrame(np.loadtxt(r'input/train/Inertial Signals/' + i),
                        index = train_labels)
    data.index.name = 'activity'
    data.reset_index(inplace=True)    
    data['subject'] = [str(int(i)) for i in subject]
    data['measurement'] = i.split('.')[0]
    
    cols = list(data.columns)
    cols = [cols[-1], cols[-2]] + cols[:-2]
    data = data[cols]

    data_dict[i.split('.')[0]] = data


df = pd.concat(list(data_dict.values()), axis = 0)

#save a sample
df[df.subject.isin(['1', '3','26', '27'])].to_csv('src/plots/raw_df.csv', index = False)

from tqdm import tqdm
#subject analysis
for i in tqdm(df.subject.unique()):
    sub  = df[df.subject == i].drop(columns = ['subject']).groupby(['measurement','activity']).head(1)#.reset_index()
    
    sub['dimension']   = sub.measurement.apply(lambda x: x[-7])
    sub['measurement'] = sub.measurement.apply(lambda x: x[:-8])
    sub = pd.melt(sub, 
            id_vars= ['measurement','dimension','activity'],
            value_vars= sub.columns[2:-1])
    sub.rename(columns = {"variable":'time'}, inplace=True)
    
    
    g = sns.FacetGrid(sub, 
                      col= 'measurement', 
                      row = 'dimension', 
                      hue = 'activity', 
                      margin_titles=True,
                      despine = False)   
    g.map(sns.lineplot, "time", "value", dashes = True)
    g.set_axis_labels("measurement", "")
    g.set_titles(col_template="{col_name}", 
                 row_template="{row_name}")
    g.add_legend()
    g.fig.suptitle(f'Subject - {i}', y = 1.02)


#analyse sitting and standing
subset = df[df.subject.isin(['1', '3','26', '27'])]#.drop(['subject'],1)
subset = subset[subset.activity.isin(['SITTING','STANDING'])]
subset = subset.groupby(['subject','measurement','activity']).head(1)
subset = pd.melt(subset, 
        id_vars= ['subject','measurement','activity'],
        value_vars= subset.columns[3:])
subset.rename(columns = {"variable":'time'}, inplace=True) 
g = sns.FacetGrid(subset, 
                  row = 'activity', 
                  col = 'measurement', 
                  hue = 'subject', 
                  margin_titles=True,
                  despine = False)   
g.map(sns.lineplot, "time", "value", dashes = True)
g.add_legend()
g.fig.suptitle('Sitting vs Standing - Subjects 1,3,26,27', y = 1.02)
g.savefig('src/plots/standing vs sitting/Subjects 1,3,26,27.png')



#analyse all activites
subset = df[df.subject.isin(['1'])]#.drop(['subject'],1)
#subset = subset[subset.activity.isin(['SITTING','STANDING'])]
subset = subset.groupby(['subject','measurement','activity']).head(1)
subset = pd.melt(subset, 
        id_vars= ['subject','measurement','activity'],
        value_vars= subset.columns[3:])
subset.rename(columns = {"variable":'time'}, inplace=True) 
g = sns.FacetGrid(subset, 
                  row = 'activity', 
                  col = 'measurement', 
                  hue = 'subject', 
                  margin_titles=True,
                  despine = False)   
g.map(sns.lineplot, "time", "value", dashes = True)
g.add_legend()
g.fig.suptitle('Sitting vs Standing - Subjects 1,3,26,27', y = 1.02)
g.savefig('src/plots/Subjects 1.png')
