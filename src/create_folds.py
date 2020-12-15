# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 06:23:37 2020

@author: deepu
"""

import pandas as pd
import numpy as np
import os
import torch
from sklearn.preprocessing import OneHotEncoder
import pickle

def create_input_vector(l : list):
    """
    
    Parameters
    ----------
    l : list of numpy arrays
        DESCRIPTION.
        length of list is 9 for each measurement from smart phone
        each value of list is (n,128) where n is the number of records and
        128 is the number of time steps

    Returns
    -------
    Reshaped tensor ready for feeding conv1d
    output shape is (n_records, n_channels, n_timesteps)
    
    
    eg output shape (7000,9,128) :
        for 2.56s sampled at 50Hz for 9 different measurments
    """
    arr = np.array(l)    
    arr = np.moveaxis(arr, (0,1,2) , (1,0,2))    
    return arr

if __name__ == "__main__":
    
    train = 'input/train'
    test  = 'input/test'
    
# =============================================================================
#     INPUT    
# =============================================================================
    #process train
    train_filename    = [train + r'/Inertial Signals/' + i for i in \
                         os.listdir(train + r'/Inertial Signals/')]                                    
    train_tensor      = create_input_vector([ np.loadtxt(i)  for i in \
                          train_filename])
    print("train processed. Output shape:", train_tensor.shape)
    
    #process test
    test_filename    = [test + r'/Inertial Signals/' + i for i in \
                         os.listdir(test + r'/Inertial Signals/')]                                    
    test_tensor      = create_input_vector([ np.loadtxt(i)  for i in \
                          test_filename])
    print("test processed. Output shape:",test_tensor.shape)


    #sample calculation
    m = torch.nn.Conv1d(in_channels=9,
               out_channels=1,
               kernel_size=2,
               padding=1)
    inp = torch.tensor(test_tensor[:1,:,:2])
                       #dtype= m.weight.type())
    output = m(inp.float())
    assert output[0,0,1] == ((inp * m.weight).sum() + m.bias)[0], "deepu: check calculation"
    
    #save
    torch.save(torch.tensor(train_tensor), 'input/Xtrain.pt')
    torch.save(torch.tensor(test_tensor),  'input/Xtest.pt')

# =============================================================================
#     TARGETS
# =============================================================================
    ytrain = np.loadtxt(train + '/y_train.txt',
                               dtype = int)
    ytest  = np.loadtxt(test + '/y_test.txt',
                               dtype = int) 
    
    activity_labels = pd.read_csv(r"input/activity_labels.txt", 
                                  sep = ' ',
                                  index_col=0,
                                  header=None).iloc[:,0].to_dict()
    
    ytrain = list( map( lambda x: [activity_labels[x]], ytrain) )
    ytest  = list( map( lambda x: [activity_labels[x]], ytest) )
    
    ohe = OneHotEncoder(sparse=False)
    ytrain_ohe = ohe.fit_transform(ytrain)
    ytest_ohe  = ohe.transform(ytest)
    
    #save
    torch.save(torch.tensor(ytrain_ohe),'input/ytrain.pt')
    torch.save(torch.tensor(ytest_ohe), 'input/ytest.pt')    
    with open("input/OneHotEncode.pkl", "wb") as f: 
        pickle.dump(ohe, f)
    
# =============================================================================
#     SUBJECTS
# =============================================================================
    subject_train = pd.Series(np.loadtxt(train + '/subject_train.txt',
                               dtype = int))
    subject_test  = pd.Series(np.loadtxt(test+ '/subject_test.txt',
                               dtype = int))
    
    subject_train.value_counts()
    subject_test.value_counts()
    assert len(set(subject_test).intersection(set(subject_train))) ==0,\
            "deepu: no overlap in subjects"
    
