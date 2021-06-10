from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Let's visualize some of the images
#%matplotlib inline
import matplotlib.pyplot as plt

def plot_py(train_losses, train_acc, test_losses, test_acc):
	fig, axs = plt.subplots(2,2,figsize=(15,10))
	axs[0, 0].plot(train_losses)
	axs[0, 0].set_title("Training Loss")
	axs[1, 0].plot(train_acc)
	axs[1, 0].set_title("Training Accuracy")
	axs[0, 1].plot(test_losses)
	axs[0, 1].set_title("Test Loss")
	axs[1, 1].plot(test_acc)
	axs[1, 1].set_title("Test Accuracy")