# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:35:33 2020

@author: deepu
"""

import wandb
wandb.init(project="pytorch-HumanActivity")
#wandb.login()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import logging
from src.custom_functions import HumanActivityDataset, DataLoader, get_cm
import random 
import pandas as pd

logging.basicConfig(filename='src/logs/wanb.log',
                    filemode='a', 
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# WandB – Initialize a new run
wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config
config.batch_size = 32         # input batch size for training (default: 64)
config.test_batch_size = 50  # input batch size for testing (default: 1000)
config.epochs = 500           # number of epochs to train (default: 10)
config.lr = 0.001             # learning rate (default: 0.01)
config.momentum = 0.5          # SGD momentum (default: 0.5) 
config.no_cuda = False         # disables CUDA training
config.seed = 42               # random seed (default: 42)
config.log_interval = 10     # how many batches to wait before logging training status
config.kernel_size_1   = 4
config.kernel_size_2   = 4
config.n_hidden        = 256


logging.info(\
f"""\n\nNetwork Details:

n_epochs         = {config.epochs}
batch_size_train = {config.batch_size}
batch_size_test  = {config.test_batch_size}
learning_rate    = {config.lr}
momentum         = {config.momentum }
    
Kernel Size L1   = {config.kernel_size_1} 
Kernel Size L2   = {config.kernel_size_2} 
n_hidden         = {config.n_hidden}
\n\n"""
)
    

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        
        self.kernel_size_1 = config.kernel_size_1
        self.out_channels1 = 9
        
        self.kernel_size_2 = config.kernel_size_2
        self.out_channels2 = 2
        
        self.padding       = 2
        self.n_hidden      = config.n_hidden
        
        
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
    


network = Net(config)
network.cuda()

gpu_available = "GPU available?:      " + str(torch.cuda.is_available())
using_cuda    = "Network using cuda?: " + str(next(network.parameters()).is_cuda)

print(gpu_available)
print(using_cuda)
logging.info("\n\n------------------------------------------------------")
logging.info(gpu_available)
logging.info(using_cuda)



random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)



optimizer = optim.SGD(network.parameters(), lr=config.lr,
                      momentum=config.momentum)
criterion = nn.CrossEntropyLoss()


if __name__ == '__main__':
    
    
    #create train loader
    train_dataset = HumanActivityDataset(file = 'train')  
    
    train_loader  = DataLoader(train_dataset, 
                               batch_size= config.batch_size,
                               shuffle=True,
                               num_workers=0)

    #create test loader
    test_dataset = HumanActivityDataset(file = 'test')  
    
    test_loader  = DataLoader(test_dataset, 
                               batch_size= config.test_batch_size,
                               shuffle=True,
                               num_workers=0)
    test_gen     = iter(test_loader)

    #print every 't' steps
    t = 5
    
    
    for epoch in range(config.epochs+1):  # loop over the dataset multiple times
    
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
    
    
                # WandB – wandb.log(a_dict) logs the keys and values of the dictionary passed in and associates the values with a step.
                # You can log anything by passing it to wandb.log, including histograms, custom matplotlib objects, images, video, text, tables, html, pointclouds and other 3D objects.
                # Here we use it to log test accuracy, loss and some test images (along with their true and predicted labels).
                wandb.log({
                    #"Examples": example_images,
                    "Test Accuracy": 100. * test_cm.diag().sum() / test_cm.sum(),
                    "Test Loss": test_loss,
                    "Train Loss": train_loss,
                    "Train Accuracy": 100. * cm.diag().sum() / cm.sum()})
                
  # WandB – Save the model checkpoint. This automatically saves a file to the cloud and associates it with the current run.
    torch.save(network.state_dict(), "src\saved_models\model.h5")
    wandb.save('src\saved_models\model.h5')
    
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
    
    #log performance on full test set
    network.eval()
    full_test_cm    = get_cm(test_dataset.y, network(test_dataset.X))   
    full_test_loss  = criterion(predictions, test_batch['labels']).item()
    msg             = 'Finished Training. Test Accuracy : {:.2f} Mean Loss : {:.2f}'.format( 
                      (full_test_cm.diag().sum() / full_test_cm.sum()).item(),
                      full_test_loss)
    print(msg)
    logging.info(msg)
    
    
    #analyze results
    activity_labels =pd.read_csv("input/LabelMap.csv", index_col=0)
    test_df = pd.DataFrame(full_test_cm.long().numpy(), 
                          columns = activity_labels.Activity,
                          index = activity_labels.Activity)
    test_df.to_csv('src/ConfusionMatrixTest.csv')
    


