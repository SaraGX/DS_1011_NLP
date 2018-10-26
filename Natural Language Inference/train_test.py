import pandas as pd
import numpy as np
import string
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    result = []
    model.eval()
    for data1, data2, lengths1, lengths2, labels in loader:
        data_batch1, data_batch2, lengths_batch1, lengths_batch2, label_batch = data1, data2, lengths1, lengths2, labels
        outputs = F.softmax(model(data_batch1, lengths_batch1, data_batch2, lengths_batch2), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
        result.append([labels, predicted])
    return (100 * correct / total) , result  


def train_model(model, train_loader, val_loader, learning_rate, num_epochs, model_name, annealing_lr, model_del, print_):
    print(model_name)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable parameters:%d'%pytorch_total_params)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if annealing_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    train_loss = []
    train_acc = []
    val_acc = []
    result = []
    for epoch in range(num_epochs):
        if annealing_lr:
            scheduler.step()
        for i, (data1, data2, lengths1, lengths2, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            # Forward pass
            outputs = model(data1, lengths1, data2, lengths2)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            train_loss.append(loss)
            
            # validate every 1000 iterations
            if i > 0 and i % 625 == 0:
                train_acc.append(test_model(train_loader, model)[0])
                val_accuracy, predicted = test_model(val_loader, model)
                result.append(predicted)
                val_acc.append(val_accuracy)
                if print_:
                    print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format( 
                       epoch+1, num_epochs, i+1, len(train_loader), val_accuracy))          
    
    print("Val Accuracy:{0:.3}%".format(test_model(val_loader, model)[0]))
    if model_del:
        del(model)
    else: 
        pkl.dump(model, open('models/'+str(model_name)+'.sav', 'wb'))
    return train_loss, train_acc, val_acc, result 