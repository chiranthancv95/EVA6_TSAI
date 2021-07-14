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

  # def train(model, device, train_loader, optimizer, scheduler, criterion):
  #     '''
  #     Trainer function
  #     '''


  #     model.train()
  #     pbar = tqdm(train_loader)

  #     correct = 0
  #     processed = 0
  #     train_loss = 0

  #     lambda_l1 = 0.001
  #     for batch_idx, (data, target) in enumerate(pbar):
  #         # get samples
  #         data, target = data["image"].to(device), target.to(device)

  #         # Init
  #         optimizer.zero_grad()

  #         # Predict
  #         y_pred = model(data)

  #         # Calculate loss
  #         loss = criterion(y_pred, target)

  #         train_loss += loss.item()

  #         # Backpropagation
  #         loss.backward()

  #         optimizer.step()
  #         scheduler.step()

  #         pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
  #         correct += pred.eq(target.view_as(pred)).sum().item()
  #         processed += len(data)

  #         pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
  #     train_acc = (100 * correct / processed)
  #     return train_acc, train_loss



def train(model, device, train_loader, optimizer, scheduler, criterion):

    model.train()
    pbar = tqdm(train_loader)

    correct = 0
    processed = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)
        lambda_l1 = 0.001
        #L1 Regularization
        if lambda_l1 > 0:
          l1 = 0
          for p in model.parameters():
            l1 = l1 + p.abs().sum()
          loss = loss + lambda_l1*l1

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