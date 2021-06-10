from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torchsummary import summary

# Let's visualize some of the images
#%matplotlib inline
import matplotlib.pyplot as plt
import argparse

from torch.optim.lr_scheduler import StepLR,OneCycleLR

from train import *
from test import *
from model import *
from plotter import *
from data import *
# from model_group_norm import *
# from model_layer_norm import *

from parser_args import norm, epochs



SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)



# train dataloader
train_loader = load_train()

# test dataloader
test_loader = load_test()

# Printing the summary of the model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using: ",device)

if norm == 'bn':
	print("Loading Batchnorm Model")
	model = Net().to(device)
elif norm == 'group':
	print("Loading Group Model")
	model = Net_group_norm().to(device)

elif norm == 'layer':
	print("Loading layer Model")
	model = Net_layer_norm().to(device)

#model = Net().to(device)
#model = Net_group_norm().to(device)
#model = Net_layer_norm().to(device)

model.apply(weights_init)
summary(model, input_size=(1, 28, 28))

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

scheduler = OneCycleLR(optimizer, max_lr=0.020,epochs=20,steps_per_epoch=len(train_loader))


for epoch in range(epochs):
    print("EPOCH:", epoch+1)
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)