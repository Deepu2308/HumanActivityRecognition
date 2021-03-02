# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:35:33 2020

@author: deepu
"""

import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import pandas as pd 
import logging
#from src.utilities import get_cm, HumanActivityDataset, DataLoader
#from src.models import CNN
from utilities import get_cm, HumanActivityDataset, DataLoader
from models import CNN


logging.basicConfig(filename='src/logs/nn_log.log',
                    filemode='a', 
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# =============================================================================
# SET TRAINING PARAMETER
# =============================================================================
n_epochs = 5000
batch_size_train = 32
batch_size_test  = 1000
learning_rate = 0.01
momentum = 0.5
weight_decay = .001

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)




network = CNN()
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
weight_decay     = {weight_decay}
Optimizer        = SGD

\n\n"""
)
logging.info(f'{network} \n\n')

optimizer = optim.SGD(network.parameters(), 
                      lr=learning_rate,
                      momentum=momentum,
                      weight_decay=weight_decay)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma=0.5, last_epoch=-1, verbose=False)
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

        #decay learning rate every 100 epochs            
        scheduler.step()

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
                cm       = get_cm(labels, outputs)
                test_cm  = get_cm(test_batch['labels'], predictions)
                test_acc = cm.diag().sum() / cm.sum()
                
                #display message every t steps and log every 10t steps
                msg = "Epoch : {} loss : {:.5f} test_loss: {:.5f} \t acc: {:.3f} test_acc: {:.3f}".format(epoch,
                       train_loss,
                       test_loss.item(),
                       test_acc,
                       test_cm.diag().sum() / test_cm.sum()
                       )
                
                print(msg)
                if epoch % (t*20) == 0: logging.info(msg)
    
    #log last available results
    logging.info(msg)
    
    #log performance on full test set
    network.eval()
    
    #inference loader
    inference_loader  = DataLoader(test_dataset, 
                               batch_size= 10,
                               shuffle=True,
                               num_workers=0)
    inference_gen     = iter(inference_loader)
    predictions, true_labels, full_test_loss = [],[],0
    for _,batch in enumerate(inference_gen):
        inputs, labels = batch['inputs'],batch['labels']
        pred = network(inputs)
        
        predictions.append(pred.cpu().detach().numpy())
        true_labels.append(labels.cpu().detach().numpy().ravel())
        full_test_loss += criterion(pred, labels).item()
        
    predictions = np.vstack(predictions)
    true_labels = np.hstack(true_labels).ravel().reshape((-1,1))
    
    full_test_cm    = get_cm(torch.Tensor(true_labels),
                             torch.Tensor(predictions) )
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
    test_df.to_csv('src/ConfusionMatrixTest__lr_scheduler.csv')