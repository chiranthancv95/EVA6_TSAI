from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torchsummary import summary

# CUDA?
cuda = torch.cuda.is_available()

def calculate_mean_std(self):
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        mean = train_set.data.mean(axis=(0,1,2))/255
        std = train_set.data.std(axis=(0,1,2))/255
        return mean, std

def load_train():
    '''
    Function to load train dataset and apply transforms
    '''

    # train_transforms = A.Compose({
    #       A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
    #       A.HorizontalFlip(),
    #       A.ShiftScaleRotate(),
    #       A.CoarseDropout(1, 16, 16, 1, 16, 16,fill_value=0.473363, mask_fill_value=None),
    #       A.ToGray(),
    #       A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
		  # ToTensorV2(),
    #   })

    #A.PadIfNeeded(min_height=40, min_width=40, always_apply=True)

    train_transforms = A.Compose({    A.RandomCrop(width=32, height=32,p=1),
                                      A.HorizontalFlip(p=1),
                                      A.CoarseDropout(max_holes=3,min_holes = 1, max_height=8, max_width=8, p=0.8,fill_value=tuple([x * 255.0 for x in mean]),
                                      min_height=8, min_width=8),
                                      A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261),always_apply=True),
                                      ToTensorV2()
                                    })

	trainset = datasets.CIFAR10(root='./data', train=True,
                        download=True, transform=train_transforms)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=False, batch_size=512, num_workers=1, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    return train_loader



def load_test():
    '''
    Function to load test dataset and apply transforms
    '''


    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    testset = datasets.CIFAR10(root='./data', download=True, transform=test_transform)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=False, batch_size=512, num_workers=1, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return test_loader