import pandas as pd
import numpy as np
import string
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle as pkl
import random

import os, re, csv, math, codecs
import io



class RNN_Encoder(nn.Module):
    def __init__(self, hidden_size, fc_dim, dropout, dropout_fc, num_layers = 1, num_classes = 3, emb_size = 300):
        # RNN Accepts the following hyperparams:    
        # hidden_size: Hidden Size of layer in RNN for premise and hypothesis, [hidden_size1, hidden_size2]
        # fc_dim: Hidden Size of layer in fully connected layer
        # dropout: dropout rate for RNN
        # dropout_fc: dropout rate for fully connected layer
        # num_layers: number of layers in RNN
        # emb_size: Embedding Size
        
        super(RNN_Encoder, self).__init__()
        self.num_layers, self.dropout, self.emb_size, self.num_classes, self.fc_dim = num_layers, dropout, emb_size, num_classes, fc_dim
        self.hidden_size1, self.hidden_size2=  hidden_size
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(id2vector)).float()
        self.rnn1 = nn.GRU(emb_size, self.hidden_size1, num_layers, self.dropout , batch_first=True, bidirectional = True)
        self.rnn2 = nn.GRU(emb_size, self.hidden_size2, num_layers, self.dropout, batch_first=True, bidirectional=True)
        self.fc_layer = nn.Sequential(nn.Linear(self.hidden_size1 + self.hidden_size2, self.fc_dim),
                                      nn.ReLU(inplace=True), 
                                      nn.Dropout(dropout_fc),
                                      nn.Linear(self.fc_dim, self.num_classes))

    def init_hidden(self, batch_size, hidden_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.randn(self.num_layers * 2, batch_size, hidden_size)
        return hidden

    def forward(self, x1, lengths1, x2, lengths2):
        
        batch_size1, seq_len1 = x1.size()
        batch_size2, seq_len2 = x2.size()
        
        # reset hidden state
        self.hidden1 = self.init_hidden(batch_size1, self.hidden_size1)
        self.hidden2 = self.init_hidden(batch_size2, self.hidden_size2)

        # get embedding of characters
        embed1 = self.embedding(x1)   
        embed2 = self.embedding(x2)
        
        #sort by length
        sent_len1, idx_sort1 = np.sort(lengths1)[::-1], np.argsort(-lengths1)
        sent_len2, idx_sort2 = np.sort(lengths2)[::-1], np.argsort(-lengths2)
        embed1= torch.index_select(embed1, 0, idx_sort1)
        embed2= torch.index_select(embed2, 0, idx_sort2)
        
        # pack padded sequence
        embed_packed1 = torch.nn.utils.rnn.pack_padded_sequence(embed1, sent_len1, batch_first=True)
        embed_packed2 = torch.nn.utils.rnn.pack_padded_sequence(embed2, sent_len2, batch_first=True)

        # fprop though RNN
        _, self.hidden1 = self.rnn1(embed_packed1, self.hidden1)
        _, self.hidden2 = self.rnn2(embed_packed2, self.hidden2)
        
        # Un-sort by length
        idx_unsort1 = np.argsort(idx_sort1)
        idx_unsort2 = np.argsort(idx_sort2)
        
        self.hidden1 = torch.index_select(self.hidden1, 1, idx_unsort1)
        self.hidden2 = torch.index_select(self.hidden2, 1, idx_unsort2)

        # sum hidden activations of RNN across time
        hn1 = torch.sum(self.hidden1 , dim=0)
        hn2 = torch.sum(self.hidden2 , dim=0)
        
        #concatenate represententations 
        hidden_out = torch.cat((hn1, hn2), dim = 1)
        
        # feed into 2 fully-connected layers
        logits = self.fc_layer(hidden_out)

        return logits

# differnt methods for interacting the two encoded sentences(element-wise multiplication)
class RNN_Encoder_element_wise(nn.Module):
    def __init__(self, hidden_size, fc_dim, dropout, dropout_fc, num_layers = 1, num_classes = 3, emb_size = 300):
        # RNN Accepts the following hyperparams:    
        # hidden_size: Hidden Size of layer in RNN for premise and hypothesis, hidden_size1 = hidden_size2
        # fc_dim: Hidden Size of layer in fully connected layer
        # dropout: dropout rate for RNN
        # dropout_fc: dropout rate for fully connected layer
        # num_layers: number of layers in RNN
        # emb_size: Embedding Size
        
        super(RNN_Encoder_element_wise, self).__init__()
        self.num_layers, self.dropout, self.emb_size, self.num_classes, self.fc_dim = num_layers, dropout, emb_size, num_classes, fc_dim
        self.hidden_size=  hidden_size
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(id2vector)).float()
        self.rnn1 = nn.GRU(emb_size, self.hidden_size, num_layers, self.dropout , batch_first=True, bidirectional = True)
        self.rnn2 = nn.GRU(emb_size, self.hidden_size, num_layers, self.dropout, batch_first=True, bidirectional=True)
        self.fc_layer = nn.Sequential(nn.Linear(self.hidden_size, self.fc_dim),
                                      nn.ReLU(inplace=True), 
                                      nn.Dropout(dropout_fc),
                                      nn.Linear(self.fc_dim, self.num_classes))

    def init_hidden(self, batch_size, hidden_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.randn(self.num_layers * 2, batch_size, hidden_size)
        return hidden

    def forward(self, x1, lengths1, x2, lengths2):
        
        batch_size1, seq_len1 = x1.size()
        batch_size2, seq_len2 = x2.size()
        
        # reset hidden state
        self.hidden1 = self.init_hidden(batch_size1, self.hidden_size)
        self.hidden2 = self.init_hidden(batch_size2, self.hidden_size)

        # get embedding of characters
        embed1 = self.embedding(x1)   
        embed2 = self.embedding(x2)
        
        #sort by length
        sent_len1, idx_sort1 = np.sort(lengths1)[::-1], np.argsort(-lengths1)
        sent_len2, idx_sort2 = np.sort(lengths2)[::-1], np.argsort(-lengths2)
        embed1= torch.index_select(embed1, 0, idx_sort1)
        embed2= torch.index_select(embed2, 0, idx_sort2)
        
        # pack padded sequence
        embed_packed1 = torch.nn.utils.rnn.pack_padded_sequence(embed1, sent_len1, batch_first=True)
        embed_packed2 = torch.nn.utils.rnn.pack_padded_sequence(embed2, sent_len2, batch_first=True)

        # fprop though RNN
        _, self.hidden1 = self.rnn1(embed_packed1, self.hidden1)
        _, self.hidden2 = self.rnn2(embed_packed2, self.hidden2)
        
        # Un-sort by length
        idx_unsort1 = np.argsort(idx_sort1)
        idx_unsort2 = np.argsort(idx_sort2)
        
        self.hidden1 = torch.index_select(self.hidden1, 1, idx_unsort1)
        self.hidden2 = torch.index_select(self.hidden2, 1, idx_unsort2)

        # sum hidden activations of RNN across time
        hn1 = torch.sum(self.hidden1 , dim=0)
        hn2 = torch.sum(self.hidden2 , dim=0)
        
        #concatenate represententations 
        hidden_out = hn1 * hn2
        
        # feed into 2 fully-connected layers
        logits = self.fc_layer(hidden_out)

        return logits

class CNN_Encoder(nn.Module):
    def __init__(self, kernel_size, kernel_pad, hidden_size, fc_dim, dropout_fc,num_layers = 2, num_classes = 3, emb_size=300):
        # CNN Accepts the following hyperparams:    
        # kernel_size
        # kernel_pad
        # hidden_size: Hidden Size of layer in CNN for premise and hypothesis, [hidden_size1, hidden_size2]
        # fc_dim: Hidden Size of layer in fully connected layer
        # dropout_fc: dropout rate for fully connected layer
        # num_layers: number of layers in CNN
        # emb_size: Embedding Size
        
        super(CNN_Encoder, self).__init__()
        self.num_layers, self.emb_size, self.num_classes, self.fc_dim = num_layers,  emb_size, num_classes, fc_dim
        self.kernel_size, self.kernel_pad = kernel_size, kernel_pad
        self.hidden_size1, self.hidden_size2 =  hidden_size
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(id2vector)).float()
        


        self.conv1_1 = nn.Conv1d(emb_size, self.hidden_size1, kernel_size = kernel_size , padding = kernel_pad )
        self.conv1_2 = nn.Conv1d(self.hidden_size1, self.hidden_size1, kernel_size = kernel_size, padding = kernel_pad )
        self.conv2_1 = nn.Conv1d(emb_size, self.hidden_size2, kernel_size = kernel_size, padding = kernel_pad )
        self.conv2_2 = nn.Conv1d(self.hidden_size2, self.hidden_size2, kernel_size = kernel_size, padding = kernel_pad )
        
        self.maxpool1 = nn.MaxPool1d(MAX_SENTENCE1_LENGTH)
        self.maxpool2 = nn.MaxPool1d(MAX_SENTENCE2_LENGTH)
        
        self.fc_layer = nn.Sequential(nn.Linear(self.hidden_size1 + self.hidden_size2, self.fc_dim),
                                      nn.ReLU(inplace=True), 
                                      nn.Dropout(dropout_fc),
                                      nn.Linear(self.fc_dim, self.num_classes))


    def forward(self, x1, lengths1, x2, lengths2):
        batch_size1, seq_len1 = x1.size()
        batch_size2, seq_len2 = x2.size()
        
        # get embedding of characters
        embed1 = self.embedding(x1)  
        embed2 = self.embedding(x2)
                
        hidden1 = self.conv1_1(embed1.transpose(1,2)).transpose(1,2)
        hidden1 = F.relu(hidden1.contiguous().view(-1, hidden1.size(-1))).view(batch_size1, seq_len1, hidden1.size(-1))
        hidden1 = self.conv1_2(hidden1.transpose(1,2)).transpose(1,2)
        hidden1 = F.relu(hidden1.contiguous().view(-1, hidden1.size(-1))).view(batch_size1, seq_len1, hidden1.size(-1))
        hidden1 = self.maxpool1(hidden1.transpose(1,2))
        
        hidden2 = self.conv2_1(embed2.transpose(1,2)).transpose(1,2)
        hidden2 = F.relu(hidden2.contiguous().view(-1, hidden2.size(-1))).view(batch_size2, seq_len2, hidden2.size(-1))
        hidden2 = self.conv2_2(hidden2.transpose(1,2)).transpose(1,2)
        hidden2 = F.relu(hidden2.contiguous().view(-1, hidden2.size(-1))).view(batch_size2, seq_len2, hidden2.size(-1))   
        hidden2 = self.maxpool2(hidden2.transpose(1,2))
        
        #concatenate represententations 
        hidden_out = torch.cat((hidden1, hidden2), dim = 1)
 
        # sum hidden activationsacross time
        hidden_out = torch.sum(hidden_out, dim = 2)
        
        # feed into 2 fully-connected layers
        logits = self.fc_layer(hidden_out)

        return logits
    
    
class CNN_Encoder_element_wise(nn.Module):
    def __init__(self, kernel_size, kernel_pad, hidden_size, fc_dim, dropout_fc,num_layers = 2, num_classes = 3, emb_size=300):
        # CNN Accepts the following hyperparams:    
        # kernel_size
        # kernel_pad
        # hidden_size: Hidden Size of layer in CNN for premise and hypothesis, hidden_size1 = hidden_size2
        # fc_dim: Hidden Size of layer in fully connected layer
        # dropout_fc: dropout rate for fully connected layer
        # num_layers: number of layers in CNN
        # emb_size: Embedding Size
        
        super(CNN_Encoder_element_wise, self).__init__()
        self.num_layers, self.emb_size, self.num_classes, self.fc_dim = num_layers,  emb_size, num_classes, fc_dim
        self.kernel_size, self.kernel_pad = kernel_size, kernel_pad
        self.hidden_size =  hidden_size
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(id2vector)).float()
        
        self.conv1_1 = nn.Conv1d(emb_size, self.hidden_size, kernel_size = kernel_size , padding = kernel_pad )
        self.conv1_2 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size = kernel_size, padding = kernel_pad )
        self.conv2_1 = nn.Conv1d(emb_size, self.hidden_size, kernel_size = kernel_size, padding = kernel_pad )
        self.conv2_2 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size = kernel_size, padding = kernel_pad )
        
        self.maxpool1 = nn.MaxPool1d(MAX_SENTENCE1_LENGTH)
        self.maxpool2 = nn.MaxPool1d(MAX_SENTENCE2_LENGTH)
        
        self.fc_layer = nn.Sequential(nn.Linear(self.hidden_size, self.fc_dim),
                                      nn.ReLU(inplace=True), 
                                      nn.Dropout(dropout_fc),
                                      nn.Linear(self.fc_dim, self.num_classes))

    def forward(self, x1, lengths1, x2, lengths2):
        batch_size1, seq_len1 = x1.size()
        batch_size2, seq_len2 = x2.size()
        
        # get embedding of characters
        embed1 = self.embedding(x1)  
        embed2 = self.embedding(x2)
                
        hidden1 = self.conv1_1(embed1.transpose(1,2)).transpose(1,2)
        hidden1 = F.relu(hidden1.contiguous().view(-1, hidden1.size(-1))).view(batch_size1, seq_len1, hidden1.size(-1))
        hidden1 = self.conv1_2(hidden1.transpose(1,2)).transpose(1,2)
        hidden1 = F.relu(hidden1.contiguous().view(-1, hidden1.size(-1))).view(batch_size1, seq_len1, hidden1.size(-1))
        hidden1 = self.maxpool1(hidden1.transpose(1,2))
        
        hidden2 = self.conv2_1(embed2.transpose(1,2)).transpose(1,2)
        hidden2 = F.relu(hidden2.contiguous().view(-1, hidden2.size(-1))).view(batch_size2, seq_len2, hidden2.size(-1))
        hidden2 = self.conv2_2(hidden2.transpose(1,2)).transpose(1,2)
        hidden2 = F.relu(hidden2.contiguous().view(-1, hidden2.size(-1))).view(batch_size2, seq_len2, hidden2.size(-1))   
        hidden2 = self.maxpool2(hidden2.transpose(1,2))
        
        #concatenate represententations 
        hidden_out = hidden1 * hidden2
 
        # sum hidden activations across time
        hidden_out = torch.sum(hidden_out, dim = 2)
        
        # feed into 2 fully-connected layers
        logits = self.fc_layer(hidden_out)

        return logits   
