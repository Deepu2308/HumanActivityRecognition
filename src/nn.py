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
import logging
from src.custom_functions import get_cm

logging.basicConfig(filename='src/logs/app.log',
                    filemode='a', 
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.kernel_size_1 = 4
        self.out_channels1 = 9
        
        self.kernel_size_2 = 4
        self.out_channels2 = 2
        
        self.padding       = 2
        self.n_hidden      = 2048
        
        
        self.conv1 =  nn.Conv1d(in_channels     = 9,
                                out_channels    = self.out_channels1,
                                kernel_size     = self.kernel_size_1,
                                padding         = self.padding)

        self.conv2 = nn.Conv1d(in_channels  = self.out_channels1,
                               out_channels = self.out_channels2,
                               kernel_size  = self.kernel_size_2,
                               padding      = self.padding)
        
        self.conv2_drop = nn.Dropout()
        self.fc1 = nn.Linear(64, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, 6)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    


network = Net()
network.cuda()

gpu_available = "GPU available?:      " + str(torch.cuda.is_available())
using_cuda    = "Network using cuda?: " + str(next(network.parameters()).is_cuda)

print(gpu_available)
print(using_cuda)
logging.info(gpu_available)
logging.info(using_cuda)

logging.info("Network using Kernel Size: L1:{} L2:{} n_hidden:{}"\
             .format(network.kernel_size_1, 
                     network.kernel_size_2,
                     network.n_hidden)
)



optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    
    Xtrain = torch.load('input/Xtrain.pt').float().cuda()
    ytrain = torch.load('input/ytrain.pt').cuda()

    Xtest  = torch.load('input/Xtest.pt').float().cuda()    
    ytest  = torch.load('input/ytest.pt').cuda()
    
    indices = np.array(range(Xtrain.shape[0]))
    n_mini_batch = 10 
    for epoch in range(5000):  # loop over the dataset multiple times
    
        network.train() 
        kf = StratifiedKFold(n_splits=n_mini_batch, shuffle= True)
        kf.get_n_splits(indices)
        
        train_loss = 0.0
        for i in kf.split(indices, y = ytrain.cpu()):
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
                
                cm      = get_cm(ytrain, network(Xtrain))
                test_cm = get_cm(ytest, predictions)
                
                msg = "Epoch : {} loss : {:.2f} test_loss: {:.2f} \t acc: {:.2f} test_acc: {:.2f}".format(epoch,
                       train_loss,
                       test_loss.item(),
                       cm.diag().sum() / cm.sum(),
                       test_cm.diag().sum() / test_cm.sum()
                       )
                logging.info(msg)
                print(msg)
    
    print('Finished Training')
    
    #save model
    if False:
        torch.save({
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
    
        },
        r"src/saved_models/model1.pkl")
    
        #load example
        checkpoint = torch.load(r"src/saved_models/model1.pkl")
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    
    #analyze results
    activity_labels =pd.read_csv("input/LabelMap.csv", index_col=0)
    test_df = pd.DataFrame(test_cm.long().numpy(), 
                          columns = activity_labels.Activity,
                          index = activity_labels.Activity)
    test_df.to_csv('src/ConfusionMatrixTest.csv')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
