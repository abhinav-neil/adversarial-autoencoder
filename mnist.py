################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2022
# Date Created: 2022-11-25
################################################################################

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


def mnist(root="../data", batch_size=64, num_workers=4, download=True):
    """
    Returns the data loader for the training set of MNIST dataset.
    Inputs:
        root - Directory in which the MNIST dataset should be downloaded. It is better to
               use the same directory as the part1 of the assignment to prevent duplicate
               downloads.
        batch_size - Batch size to use for the data loaders
        num_workers - Number of workers to use in the data loaders.
        download - If True, MNIST is downloaded if it cannot be found in the specified
                   root directory.
    """

    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root, train=True, download=download,
                                   transform=data_transforms)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    return train_loader
