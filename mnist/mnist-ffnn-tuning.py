#region imports
import argparse
import os.path as osp
from time import time
from datetime import datetime, timedelta
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as T
from torchvision.datasets import MNIST

from ray import tune
from ray.tune import CLIReporter, JupyterNotebookReporter
from ray.tune.schedulers import ASHAScheduler
#endregion

#region definitions
class FFNN(nn.Module):
    def __init__(self, in_shape, out_shape, p_d1=0.5, p_d2=0.4, h1=64, h2=32):
        super().__init__()
        fc1 = nn.Linear(in_shape, h1)
        a1  = nn.ReLU()
        d1  = nn.Dropout(p=p_d1)
        fc2 = nn.Linear(h1, h2)
        a2  = nn.ReLU()
        d2  = nn.Dropout(p=p_d2)
        fc3 = nn.Linear(h2, out_shape)
        
        # not applying log_softmax here, as it is applied later in 
        # the torch CCE loss
        
        self.nn = nn.Sequential(fc1, a1, d1, fc2, a2, d2, fc3)

    def forward(self, x):
        x = self.nn(x)
        return x
    
def train_mnist(config, epochs, checkpoint_dir=None, data_dir=None):
    # create model
    model = FFNN(784, 10, 
                 p_d1=config['p_d1'], 
                 p_d2=config['p_d2'], 
                 h1=config['h1'], 
                 h2=config['h2'])
    
    # load data and make a validation split
    transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,),(0.5)), 
                            T.Lambda(lambda x: torch.flatten(x))])
    dataset_train = MNIST(root='/data/', transform=transforms, train=True)

    train_samples = int(len(dataset_train) * 0.8)
    train_subset, val_subset = random_split(dataset_train,
                                           [train_samples, 
                                            len(dataset_train) - train_samples])
    # create dataloaders
    train_args = {'dataset':train_subset, 
                  'batch_size':config['batch_size'], 
                  'shuffle':True, 
                  'num_workers':8, 
                  'pin_memory':True}
    dataloader_train = torch.utils.data.DataLoader(**train_args)
    val_args  = {'dataset':val_subset, 
                  'batch_size':len(val_subset), 
                  'shuffle':False, 
                  'num_workers':8}
    dataloader_val  = torch.utils.data.DataLoader(**val_args) 
    
    # choose computation host device
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    model.to(device)
    
    
    optimiser = torch.optim.SGD(params=model.parameters(), lr=config['lr'], momentum=0.9)
    f_loss = nn.CrossEntropyLoss()
    
    # training loop
    for n in range(epochs):
        total_loss = 0.0
        # optimisation
        model.train()
        for idx, (X, y) in enumerate(dataloader_train):
            X, y = X.to(device), y.to(device)
            optimiser.zero_grad()
            y_pred = model(X)
            loss = f_loss(y_pred, y)
            loss.backward()
            total_loss += loss.detach().cpu().item() / len(y) # normalise for batch size
            optimiser.step()
            
        # validation set metrics
        predictions, targets, val_losses = [], [], []
        model.eval()
        # we are adding the metrics tensor for each batch to a list,
        # then concatenating at the end to make one tensor with all samples
        for idx, (X, y) in enumerate(dataloader_val):
            with torch.no_grad():
                y_pred = model(X)
                predictions.append(y_pred.detach())
                targets.append(y)
                val_losses.append(f_loss(y_pred, y).cpu().item())

        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        predictions = torch.argmax(F.log_softmax(predictions, dim=1),dim=1)
        corrects = (predictions == targets).sum().item()
        wrongs = len(targets) - corrects
        val_accuracy = corrects / len(targets)
        val_loss = sum(val_losses) / float(len(val_losses))
        
        # save checkpoint
        with tune.checkpoint_dir(n) as checkpoint_dir:
            path = osp.join(checkpoint_dir, 'checkpoint')
            torch.save((model.state_dict(), optimiser.state_dict()), path)
            
        # report metric values back to main scheduler
        tune.report(loss=val_loss, accuracy=val_accuracy)
        
def test_accuracy(model, device='cpu'):
    transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,),(0.5)), 
                            T.Lambda(lambda x: torch.flatten(x))])
    dataset_test  = MNIST(root='/data/', transform=transforms, train=False)
    test_args  = {'dataset':dataset_test, 
                  'batch_size':len(dataset_test), 
                  'shuffle':False, 
                  'num_workers':8}
    dataloader_test  = torch.utils.data.DataLoader(**test_args) 
    
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for (X, y) in dataloader_test:
            y_pred = model(X)
            predictions.append(y_pred.detach())
            targets.append(y)

        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        predictions = torch.argmax(F.log_softmax(predictions, dim=1),dim=1)
        corrects = (predictions == targets).sum().item()
        wrongs = len(targets) - corrects
        test_accuracy = corrects / len(targets)
        
    return  test_accuracy
def main(max_epochs=20, num_trials=30, is_notebook=True):
    config = {'lr':tune.loguniform(1e-3, 1e-1), 
              'batch_size':tune.choice([32, 64, 128, 256, 512]), 
              'p_d1':tune.uniform(0.2, 0.9), 
              'p_d2':tune.uniform(0.2, 0.9), 
              'h1':tune.choice([64, 256, 512, 1024]), 
              'h2':tune.choice([32, 64, 256, 512])}


    scheduler = ASHAScheduler(metric='loss', 
                            mode='min',
                            max_t=20, 
                            grace_period=2, 
                            reduction_factor=2)
    
    metric_columns = ['loss', 'accuracy', 'training_iteration']
    if is_notebook:
        reporter = JupyterNotebookReporter(overwrite=True, max_progress_rows=num_trials, metric_columns=metric_columns)
    else:
        reporter = CLIReporter(metric_columns=metric_columns)
    #reporter  = CLIReporter(metric_columns=['loss', 'accuracy', 'training_iteration'])

    resources = {'cpu':2} 
    if torch.cuda.is_available():
        resources['gpu'] = 0.5

    result = tune.run(partial(train_mnist, epochs=max_epochs),
                      resources_per_trial=resources, 
                      config=config, 
                      num_samples=num_trials, 
                      scheduler=scheduler, 
                      progress_reporter=reporter)

    best_trial = result.get_best_trial('loss', 'min', 'last')
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]))
    best_trained_model = FFNN(784, 10, 
                              p_d1=best_trial.config['p_d1'], 
                              p_d2=best_trial.config['p_d2'], 
                              h1=best_trial.config['h1'], 
                              h2=best_trial.config['h2'])

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimiser_state = torch.load(osp.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
    test_acc = test_accuracy(best_trained_model)
    print("Best trial test set accuracy: {}".format(test_acc))
    
    return best_trial
#endregion



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run some parameter tuning on a Feed Forward NN trained on the MNIST dataset')
    parser.add_argument('--n_epochs', help='max number of training iterations per trial', 
                        type=int, default=20)
    parser.add_argument('--n_trials', help='number of distinct parameter configurations to try out', 
                        type=int, default=20)
    args = parser.parse_args()
    main(max_epochs=args.n_epochs, num_trials=args.n_trials, is_notebook=False)
    
    