from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.nn.init import kaiming_uniform_

dropout_value = 0.03
class Net_group_norm(nn.Module):
    def __init__(self):
        super(Net_group_norm, self).__init__()
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False), # Input 28x28 output 26x26 RF : 3x3
            nn.ReLU(),
            #nn.BatchNorm2d(8),
            nn.GroupNorm(4, 8),

            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False), # Input 26x26 output 24x24 RF : 5x5
            nn.ReLU(),
            # nn.BatchNorm2d(16),
            nn.GroupNorm(8, 16),
            nn.Dropout(dropout_value)
        ) 

        #Transition Block
        self.trans1 = nn.Sequential(
            
            nn.MaxPool2d(2, 2), # output_size = 12
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False)  # Input 12x12 output 12x12 RF : 6x6
        ) # output_size = 24
        

        # CONVOLUTION BLOCK 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),   # Input 12x12 output 10x10 RF : 6x6
            nn.ReLU(),            
            #nn.BatchNorm2d(10),
            nn.GroupNorm(5, 10),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),  # Input 10x10 output 8x8 RF : 10x10
            nn.ReLU(),            
            #nn.BatchNorm2d(16),
            nn.GroupNorm(8, 16),
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=16, out_channels=18, kernel_size=(3, 3), padding=0, bias=False),  # Input 8x8 output 6x6 RF : 14x14
            nn.ReLU(),            
            #nn.BatchNorm2d(18),
            nn.GroupNorm(9, 18),
            nn.Dropout(dropout_value)

        ) 
        
        # OUTPUT BLOCK
        self.avgpool2d = nn.AvgPool2d(kernel_size=6)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=16, kernel_size=(1, 1), padding=0, bias=False), # Input 6x6 output 6x6 RF : 18x18
            nn.ReLU(),            
            #nn.BatchNorm2d(16),
            nn.GroupNorm(8, 16),
            nn.Dropout(dropout_value))

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)  # Input 6x6 output 6x6 RF : 18x18


    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.avgpool2d(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

#nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        #xavier(m.bias.data)

# model = Net().cuda()
# model.apply(weights_init)