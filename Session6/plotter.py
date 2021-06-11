from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Let's visualize some of the images
#%matplotlib inline
import matplotlib.pyplot as plt

def plot_py(train_losses, train_acc, test_losses, test_acc, exp_name):
	fig, axs = plt.subplots(2,2,figsize=(15,10))
	axs[0, 0].plot(train_losses)
	axs[0, 0].set_title("Training Loss")
	axs[1, 0].plot(train_acc)
	axs[1, 0].set_title("Training Accuracy")
	axs[0, 1].plot(test_losses)
	axs[0, 1].set_title("Test Loss")
	axs[1, 1].plot(test_acc)
	axs[1, 1].set_title("Test Accuracy")
	fig.suptitle(exp_name)



def false_plotter(false_predictions, exp_name):
    fig = plt.figure(figsize=(8,10))
    fig.tight_layout()
    for i, (img, pred, correct) in enumerate(false_predictions[:10]):
        img, pred, target = img.cpu().numpy(), pred.cpu(), correct.cpu()
        ax = fig.add_subplot(5, 2, i+1)
        ax.axis('off')
        ax.set_title(f'\nactual {target.item()}\npredicted {pred.item()}',fontsize=10)  
        ax.imshow(img.squeeze(), cmap='gray_r')
        fig.suptitle(exp_name+'_experiment')

    plt.show()