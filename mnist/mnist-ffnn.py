import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from time import time
import os


class FFNN(nn.Module):
    def __init__(self, in_shape):
        pass

    def forward(self, x):
        pass

if __name__=='__main__':
    # define dataset splits 
    dataset = torchvision.datasets.MNIST
    transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,),(0.5))])

    dataset_train = dataset(root='/data', transform=transforms, train=True)
    dataset_test  = dataset(root='/data', transform=transforms, train=False)

    # create dataloaders
    train_args = {'dataset':dataset_train, 'batch_size':64, 'shuffle':True, 'num_workers':0, 'pin_memory':True}
    dataloader_train = torch.utils.data.DataLoader(**train_args)
    test_args  = {'dataset':dataset_test, 'batch_size':len(dataset_test), 'shuffle':False, 'num_workers':8}
    dataloader_test  = torch.utils.data.DataLoader(**test_args) 

    
    batch = next(iter(dataloader_train))
