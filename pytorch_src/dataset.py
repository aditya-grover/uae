import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import os
import numpy as np

class ClassicDataset(Dataset):
    
    def __init__(self,
        x,
        y,
        transform):

        self.xy = TensorDataset(x, y)
        self.transform = transform

    def __len__(self):

        return len(self.xy) 

    def __getitem__(self, idx):

        x, y = self.xy[idx]
        if self.transform:
            x = self.transform(x)

        return x, y


def get_loaders(dataset,
    datadir,
    batch_size,
    test_batch_size,
    kwargs):
    
    
    if dataset == 'mnist':
        trva_real = datasets.MNIST(root=datadir, download=True)
        tr_real_ds, va_real_ds = random_split(trva_real, [50000, 10000])
        xtr_real = trva_real.train_data[tr_real_ds.indices].view(-1, 1, 28, 28)
        ytr_real = trva_real.train_labels[tr_real_ds.indices]
        xva_real = trva_real.train_data[va_real_ds.indices].view(-1, 1, 28, 28)
        yva_real = trva_real.train_labels[va_real_ds.indices]

        trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

        train_dataset = ClassicDataset(x=xtr_real, y=ytr_real, transform=trans)
        valid_dataset = ClassicDataset(x=xva_real, y=yva_real, transform=trans)
        test_dataset = datasets.MNIST(root=datadir, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset,
            batch_size=batch_size, shuffle=True, **kwargs)

    valid_loader = DataLoader(valid_dataset,
        batch_size=test_batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(test_dataset,
        batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, valid_loader, test_loader



def get_train_test_loaders(dataset,
    datadir,
    batch_size,
    test_batch_size,
    kwargs):
    
    
    if dataset == 'mnist':
        train_dataset = datasets.MNIST(root=datadir, download=True, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        test_dataset = datasets.MNIST(root=datadir, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset,
            batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(test_dataset,
        batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader