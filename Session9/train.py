from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR,OneCycleLR

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, scheduler, criterion):
    '''
    Trainer function
    '''
    model.train()
    pbar = tqdm(train_loader)

    correct = 0
    processed = 0
    train_loss = 0

    lambda_l1 = 0.001
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data["image"].to(device), target.to(device)

        # Init
        optimizer.zero_grad()

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)

        train_loss += loss.item()

        # Backpropagation
        loss.backward()

        optimizer.step()
        scheduler.step()

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
    train_acc = (100 * correct / processed)
    return train_acc, train_loss