from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Let's visualize some of the images
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

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



def plot_graphs(train_losses, test_losses, train_accuracy, test_accuracy):
    '''
    Function which plots graphs for train and test accuracy and loss
    '''
    sns.set(style='whitegrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (20, 10)

    # Plot the learning curve.
    fig, (plt1, plt2) = plt.subplots(1, 2)
    plt1.plot(np.array([x / 1000 for x in train_losses]), 'r', label="Training Loss")
    plt1.plot(np.array(test_losses), 'b', label="Validation Loss")
    plt2.plot(np.array(train_accuracy), 'r', label="Training Accuracy")
    plt2.plot(np.array(test_accuracy), 'b', label="Validation Accuracy")

    plt2.set_title("Training-Validation Accuracy Curve")
    plt2.set_xlabel("Epoch")
    plt2.set_ylabel("Accuracy")
    plt2.legend()
    plt1.set_title("Training-Validation Loss Curve")
    plt1.set_xlabel("Epoch")
    plt1.set_ylabel("Loss")
    plt1.legend()

    plt.show()