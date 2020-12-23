# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:53:44 2020

@author: deepu
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd 

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

def get_cm(actual, predictions):
    nb_classes = 6
    
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        
            _, preds = torch.max(predictions, 1)
            for t, p in zip(actual.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
    
    return confusion_matrix.long()



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 =  nn.Conv1d(in_channels=9,
                                      out_channels=6,
                                      kernel_size=2,
                                      padding=1)

        self.conv2 = nn.Conv1d(in_channels=6,
                                      out_channels=2,
                                      kernel_size=2,
                                      padding=1)
        
        self.conv2_drop = nn.Dropout()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 6)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    

    
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
criterion = nn.CrossEntropyLoss()

Xtrain = torch.load('input/Xtrain.pt').float()
Xtest  = torch.load('input/Xtest.pt').float()
ytrain = torch.load('input/ytrain.pt')
ytest  = torch.load('input/ytest.pt')

indices = np.array(range(Xtrain.shape[0]))
n_mini_batch = 10 
for epoch in range(5000):  # loop over the dataset multiple times

    network.train() 
    kf = StratifiedKFold(n_splits=n_mini_batch, shuffle= True)
    kf.get_n_splits(indices)
    
    train_loss = 0.0
    for i in kf.split(indices, y = ytrain):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = Xtrain[i[1]], ytrain[i[1]]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # print epoch level statistics
    if epoch % 5 == 0:
        with torch.no_grad():
            predictions = network(Xtest)
            test_loss = criterion(predictions, ytest)
            train_loss /= n_mini_batch
            
            cm     = get_cm(ytrain, network(Xtrain))
            val_cm = get_cm(ytest, predictions)
            
            print("Epoch : {} loss : {:.2f} val_loss: {:.2f} \t acc: {:.2f} val_acc: {:.2f}".format(epoch,
                   train_loss,
                   test_loss.item(),
                   cm.diag().sum() / cm.sum(),
                   val_cm.diag().sum() / val_cm.sum()
                   )
            )

print('Finished Training')

#save model
if False:
    torch.save({
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,

    },
    r"src/saved_models/first_attempt.pkl")

    #load example
    checkpoint = torch.load(r"src/saved_models/first_attempt.pkl")
    network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

#analyze results
activity_labels =pd.read_csv("input/LabelMap.csv", index_col=0)
val_df = pd.DataFrame(val_cm.long().numpy(), 
                      columns = activity_labels.Activity,
                      index = activity_labels.Activity)


