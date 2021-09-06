import torch
import torch.nn as nn
import torch.optim as optim
import attr

from abc import ABC, abstractmethod

class Trainer(ABC):
    ''' Class to inherit from for creating a training scheme when using the AdvLib benchmark. 
        Mainly for robust training.
    '''
    @abstractmethod
    def train(self):
        ''' performs training on a batch of data
            returns the loss
        '''
        pass

@attr.s
class Robust_trainer(Trainer):
    '''Performs robust training
    '''

    # get the type of optimizer and loss function to use
    optimizer:optim.Optimizer = attr.ib()
    loss:nn.Module = attr.ib() 

    def train(self, model, labels, adv_inputs, adv_outputs, adv_predictions, attack):
        
        cost = self.loss(adv_outputs, labels)
        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()
        return cost



