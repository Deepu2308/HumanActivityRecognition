# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 03:41:46 2020

@author: deepu
"""
import torch

def get_cm(actual, predictions):
    nb_classes = 6
    
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        
            _, preds = torch.max(predictions, 1)
            for t, p in zip(actual.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
    
    return confusion_matrix.long()
