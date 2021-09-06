import math
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.

    e.g use:
    
    MNISTWithIndices = dataset_with_indices(MNIST)
    dataset = MNISTWithIndices('~/datasets/mnist')

    for batch_idx, (data, target, idx) in enumerate(loader):
        print('Batch idx {}, dataset index {}'.format(
            batch_idx, idx))

    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


class CIFAR10_module(pl.LightningDataModule):
    ''' Note the Train is shuffeled but not the test
    '''

    def __init__(self, mean, std, data_dir: str = "../data" , batch_size: int = 32, num_workers=12):
        '''The mean and std used for normalization during training should be sent.
        
        '''
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = mean
        self.std = std
        # we might need individual indices
        self.cifar_with_idx = dataset_with_indices(CIFAR10)


    def prepare_data(self):
        # download
        self.cifar_with_idx(root=self.data_dir, train=True, download=True)
        self.cifar_with_idx(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        # Assign train dataset for use in dataloaders
        if stage == 'fit' or stage is None:
            train_transforms = transforms.Compose(
                                        [transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(self.mean, self.std) ])
            self.cifar10_train = self.cifar_with_idx(root=self.data_dir, train=True, download=True, transform=train_transforms)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            test_transforms = transforms.Compose(
                                        [transforms.ToTensor(),
                                        transforms.Normalize(self.mean, self.std)])
            self.cifar10_test = self.cifar_with_idx(root=self.data_dir, train=False, download=True, transform=test_transforms)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True, 
                         num_workers=self.num_workers, pin_memory=True)
                         

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, shuffle=False, 
                        num_workers=self.num_workers, pin_memory=True)
                        

