from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from tqdm import tqdm


train_losses = []
test_losses = []
train_acc = []
test_acc = []

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    false_images=[]
    false_label=[]
    correct_label=[]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            false_pred = (pred.eq(target.view_as(pred)) == False)
            false_images.append(data[false_pred])
            false_label.append(pred[false_pred])
            correct_label.append(target.view_as(pred)[false_pred])  

            false_predictions = list(zip(torch.cat(false_images),torch.cat(false_label),torch.cat(correct_label)))    
        print(f'Total false predictions are {len(false_predictions)}')



    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))
    return test_losses, test_acc, false_predictions