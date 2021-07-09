from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torchsummary import summary
from torchscan import summary

# Let's visualize some of the images
#%matplotlib inline
import matplotlib.pyplot as plt
import argparse

from torch.optim.lr_scheduler import StepLR,OneCycleLR

from train import *
from test import *
#from model import *
from plotter import *
from data import *
# from model_group_norm import *
# from model_layer_norm import *

from parser_args import norm, epochs
from custom_resnet import *


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

#Loading CustomResnet
model = CustomResNet().to(device)


#model.apply(weights_init)

#TorchSummary

summary(model, input_size=(1, 32, 32))



#torchscan for Receptive Field Calculations
summary(model, (3, 32, 32), receptive_field=True, max_depth=1)

# optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9)

# scheduler = OneCycleLR(optimizer, max_lr=0.015, epochs=60, steps_per_epoch=len(train_loader))

optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
lr_finder = LRFinder(net, optimizer, criterion, device=device)
lr_finder.range_test(train_loader, end_lr=10, num_iter=200)
lr_finder.plot()

min_loss = min(lr_finder.history['loss'])
lr_rate = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]
print("Max LR is {}".format(lr_rate))

epochs=24

optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=lr_rate/10,
                                                steps_per_epoch=len(train_loader), 
                                                epochs=24,
                                                pct_start=0.2,
                                                div_factor=10,
                                                three_phase=False, 
                                                final_div_factor=50,
                                                anneal_strategy='linear'
                                                )

train_losses = []
test_losses = []
train_accuracy = []
test_accuracy = []


for epoch in range(1, epochs + 1):
    print(f'Epoch {epoch}:')
    train_acc, train_loss = train(model, device, train_loader, optimizer,epoch,scheduler,criterion)
    train_accuracy.append(train_acc)
    train_losses.append(train_loss)

    test_acc, test_loss = test(
            model=model,
            device=device,
            test_loader=test_loader
        )


    test_accuracy_values.append(test_acc)
    test_loss_values.append(test_loss)

    plot_graphs(
        train_losses=train_loss_values,
        test_losses=test_loss_values,
        train_accuracy=train_accuracy_values,
        test_accuracy=test_accuracy_values
    )  # Plotting Graphs
