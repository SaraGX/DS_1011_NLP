import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# MAX_SENTENCE_LENGTH is derived from 99% sentence length of SNLI training set
MAX_SENTENCE1_LENGTH = 32
MAX_SENTENCE2_LENGTH = 18

class NLI_Dataset(Dataset):
    
    """
    Class that represents a train/validation dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, data_list1, data_list2, target_list):
        """
        @param data_list1: list of NLI tokens in premise
        @param data_list2: list of NLI tokens in hypothesis
        @param target_list: list of NLI targets 

        """
        self.data_list1 = data_list1
        self.data_list2 = data_list2
        self.target_list = target_list
        assert (len(self.data_list1) == len(self.target_list))
        assert (len(self.data_list2) == len(self.target_list))

    def __len__(self):
        return len(self.target_list)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """      
        token_idx1 = self.data_list1[key][:MAX_SENTENCE1_LENGTH]
        token_idx2 = self.data_list2[key][:MAX_SENTENCE2_LENGTH]
        label = self.target_list[key]
        return [token_idx1, token_idx2, len(token_idx1),len(token_idx2), label]
    
def NLI_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    data_list1 = []
    data_list2 = []
    label_list = []
    length_list1 = []
    length_list2 = []

    for datum in batch:
        label_list.append(datum[4])
        length_list1.append(datum[2])
        length_list2.append(datum[3])
    # padding
    for datum in batch:
        padded_vec1 = np.pad(np.array(datum[0]), pad_width=((0,MAX_SENTENCE1_LENGTH-datum[2])), mode="constant", constant_values=0)
        data_list1.append(padded_vec1)
        padded_vec2 = np.pad(np.array(datum[1]), pad_width=((0,MAX_SENTENCE2_LENGTH-datum[3])), mode="constant", constant_values=0)
        data_list2.append(padded_vec2)

    return [torch.from_numpy(np.array(data_list1)), torch.from_numpy(np.array(data_list2)), 
            torch.LongTensor(length_list1), torch.LongTensor(length_list2), 
            torch.LongTensor(label_list)]

