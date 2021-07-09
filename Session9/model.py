import torch.nn as nn
import torch.nn.functional as F

#depthwise separable convolution
class DepthwiseSeparable(nn.Module):
  def __init__(self, in_ch, out_ch, stride=1):
    super(DepthwiseSeparable, self).__init__()
    self.in_chan = in_ch
    self.out_chan = out_ch

    self.depthwise = nn.Sequential(
          nn.Conv2d(in_channels=self.in_chan, out_channels=self.in_chan, kernel_size=(3, 3), padding=1, stride=stride, groups=self.in_chan, bias=False),
          #pointwise
          nn.Conv2d(in_channels=self.in_chan, out_channels=self.out_chan, kernel_size=(1,1)))

  def forward(self, x):
    x = self.depthwise(x)
    return x


dropout=0.025
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),   
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.transblock1 =  nn.Sequential(
            #Stride 2 conv
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=2, bias=False), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),  

        ) 

        self.transblock2 =  nn.Sequential(
            #Stride 2 conv
            DepthwiseSeparable(64,64,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.convblock3 = nn.Sequential(
            DepthwiseSeparable(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout),  
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout), 

        ) 

        self.transblock3 =  nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation=2, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), 
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),

        ) 

        self.gap = nn.AvgPool2d(4)

    def forward(self, x):
        x =  self.transblock1(self.convblock1(x))
        x =  self.transblock2(self.convblock2(x))
        x =  self.transblock3(self.convblock3(x))
        x =  self.convblock4(x)
        x =  self.gap(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)

#depthwise separable convolution
class DepthwiseSeparable(nn.Module):
  def __init__(self, in_ch, out_ch):
    super(DepthwiseSeparable, self).__init__()
    self.in_ch = in_ch
    self.out_ch = out_ch

    self.depthwise = nn.Sequential(
          nn.Conv2d(in_channels=self.in_chan, out_channels=self.in_chan, kernel_size=(3, 3), padding=1, stride=1, groups=self.in_chan, bias=False),
          #pointwise
          nn.Conv2d(in_channels=self.in_chan, out_channels=self.out_chan, kernel_size=(1,1)))

  def forward(self, x):
    x = self.depthwise(x)
    return x