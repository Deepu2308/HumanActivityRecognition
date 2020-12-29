# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 03:41:46 2020

@author: deepu
"""
import torch
from torch.utils.data import Dataset, DataLoader

def get_cm(actual, predictions):
    nb_classes = 6
    
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        
            _, preds = torch.max(predictions, 1)
            for t, p in zip(actual.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
    
    return confusion_matrix.long()


class HumanActivityDataset(Dataset):
    """Human Activity dataset."""

    def __init__(self,  
                 root_dir = r"input/", 
                 file='train'):
        """
        Args:
            root_dir (string, optional): Directory with all the input and label files.
            file (string, optional):     'train' or 'test'
        """
            
        self.X = torch.load(root_dir + f'X{file}.pt').float().cuda()
        self.y = torch.load(root_dir + f'y{file}.pt').cuda()   
    
        self.root_dir   = root_dir
        self.file       = file

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = {'inputs': self.X[idx], 'labels': self.y[idx]}
        
        return sample
    
if __name__ == '__main__':
    
    #sample usage
    train_dataset = HumanActivityDataset(file = 'train')  
    
    train_loader  = DataLoader(train_dataset, 
                               batch_size=4,
                               shuffle=True,
                               num_workers=0)
    
    
    for i_batch, sample_batched in enumerate(train_loader):
        print(i_batch, sample_batched['inputs'].size(),
              sample_batched['labels'].size())
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    