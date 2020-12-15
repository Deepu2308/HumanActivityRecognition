# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 06:26:09 2020

@author: deepu
"""


#import requrired modules
import argparse
import os
import joblib

from sklearn import metrics
import pandas as pd

#import custom modules
import config
import model_dispatcher

def run(fold, model, target):
    
    #read the training file with folds
    df = pd.read_csv(config.TRAINING_FILE,   
                     dtype = {'fold': int})
    
    
    #assert fold is valid
    assert fold in df.fold.unique(), "Invalid fold number"
    
    #assert target is valid
    assert target in df.columns, "Invalid target column name"
    
    #assert model is available
    assert model in model_dispatcher.models.keys(), "Model not available"
    
    #train, validation set split
    df_train = df[df.fold != fold].reset_index(drop = True)
    df_valid = df[df.fold == fold].reset_index(drop = True)
    
    #create predictors
    X_train = df_train.drop(target, axis=1)
    X_valid = df_valid.drop(target, axis=1)
    
    #create targets
    y_train = df_train[target]
    y_valid = df_valid[target]
    
    #fetch model from model_dispatcher
    clf = model_dispatcher.models[model]
    
    #fit model
    clf.fit(X_train, y_train)
    
    #predict on validation set
    pred = clf.predict(X_valid)
    
    #calculate accuracy
    acc = metrics.accuracy_score(y_valid,pred)
    print("Fold: {} Accuracy: {:.2f}".format(fold, acc))    
    
    #save model (consider writing versioning code)
    joblib.dump(clf, 
                os.path.join(config.MODEL_OUTPUT, f"{model}_f{fold}.bin")
    )
    
    
if __name__ == '__main__':
    
    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",
                        const = 1,
                        type = int)    
    parser.add_argument("--target",
                        type = str)    
    parser.add_argument("--model",
                    type = str)
    args = parser.parse_args()     
    
    #run model
    run(args.fold, 
        args.model, 
        args.target
    )
    
    
    
    
    