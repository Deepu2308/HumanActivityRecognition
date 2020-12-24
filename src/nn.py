# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:35:33 2020

@author: deepu
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import pandas as pd 
import logging
from src.custom_functions import get_cm, HumanActivityDataset, DataLoader

logging.basicConfig(filename='src/logs/nn_log.log',
                    filemode='a', 
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


n_epochs = 1500
batch_size_train = 32
batch_size_test  = 1000
learning_rate = 0.001
momentum = 0.5

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
        self.n_hidden      = 256
        
        
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
logging.info("\n\n------------------------------------------------------")
logging.info(gpu_available)
logging.info(using_cuda)

logging.info(\
f"""\n\nNetwork Details:

n_epochs         = {n_epochs}
batch_size_train = {batch_size_train}
batch_size_test  = {batch_size_test}
learning_rate    = {learning_rate}
momentum         = {momentum}
    
Kernel Size L1   = {network.kernel_size_1} 
Kernel Size L2   = {network.kernel_size_2} 
n_hidden         = {network.n_hidden}
\n\n"""
)

optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    
    
    #create train loader
    train_dataset = HumanActivityDataset(file = 'train')  
    
    train_loader  = DataLoader(train_dataset, 
                               batch_size= batch_size_train,
                               shuffle=True,
                               num_workers=0)

    #create test loader
    test_dataset = HumanActivityDataset(file = 'test')  
    
    test_loader  = DataLoader(test_dataset, 
                               batch_size= batch_size_test,
                               shuffle=True,
                               num_workers=0)
    test_gen     = iter(test_loader)

    #print every 't' steps
    t = 5
    
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
    
        network.train() 
        train_loss = 0.0
        
        for i_batch, batch in enumerate(train_loader):
        
            # get the inputs and labels
            inputs, labels = batch['inputs'],batch['labels']
        
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
        # print epoch level statistics
        if epoch % t == 0:
            network.eval()
            with torch.no_grad():
                
                #get test batch
                try:
                        test_batch      = next(test_gen)
                except StopIteration:
                        test_gen        = iter(test_loader)                
                        test_batch      = next(test_gen)                                
                
                #get predictions
                predictions = network(test_batch['inputs'])
                
                #compute losses
                test_loss  = criterion(predictions, test_batch['labels'])
                train_loss /= (i_batch+1)
                
                #create confusion matrix
                cm      = get_cm(labels, outputs)
                test_cm = get_cm(test_batch['labels'], predictions)
                
                #display message every t steps and log every 10t steps
                msg = "Epoch : {} loss : {:.2f} test_loss: {:.2f} \t acc: {:.2f} test_acc: {:.2f}".format(epoch,
                       train_loss,
                       test_loss.item(),
                       cm.diag().sum() / cm.sum(),
                       test_cm.diag().sum() / test_cm.sum()
                       )
                
                print(msg)
                if epoch % (t*20) == 0: logging.info(msg)
    
    #log last available results
    logging.info(msg)
    
    #log performance on full test set
    network.eval()
    full_test_cm    = get_cm(test_dataset.y, network(test_dataset.X))   
    full_test_loss  = criterion(predictions, test_batch['labels']).item()
    msg             = 'Finished Training. Test Accuracy : {:.2f} Mean Loss : {:.2f}'.format( 
                      (full_test_cm.diag().sum() / full_test_cm.sum()).item(),
                      full_test_loss)
    print(msg)
    logging.info(msg)
    
    
    #save model
    if True:
        torch.save({
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
    
        },
        r"src/saved_models/model2.pkl")
    
        #load example
        checkpoint = torch.load(r"src/saved_models/model2.pkl")
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    
    #analyze results
    activity_labels =pd.read_csv("input/LabelMap.csv", index_col=0)
    test_df = pd.DataFrame(full_test_cm.long().numpy(), 
                          columns = activity_labels.Activity,
                          index = activity_labels.Activity)
    test_df.to_csv('src/ConfusionMatrixTest.csv')
    
    
    
    
    
    
    
    
    
    
    
    
    
    


