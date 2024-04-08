import torch
import torch.nn as nn

from . import unet



class EncoderMLP(nn.Module):
    def __init__(self, chan = [1, 3, 9], initw = False):
        super().__init__()

        self.l1 = nn.Linear(2500,500 )
        self.l2 = nn.Linear(500,100 )
        self.l3 = nn.Linear(100,1 )

        self.relu= nn.ReLU()
        self.Softmax= nn.Softmax(dim=1)
        self.sigmoid= nn.Sigmoid()

        self.Tanh=nn.Tanh()

        if initw:

          #xavier_normal
          nn.init.xavier_normal_(self.l1.weight)
          nn.init.xavier_normal_(self.l2.weight)
          nn.init.xavier_normal_(self.l3.weight)

    def forward(self, x):

      x = x.view(x.shape[0], -1)
      x = self.relu(self.l1(x))
      x = self.relu(self.l2(x))
      x = self.l3(x)

      return x
    
class EncoderCNN(nn.Module):
    def __init__(self, in_channels, n_iter, initw = False):
        super().__init__()

        self.convs = nn.ModuleList()

        self.in_channels = in_channels
        self.n_iter = n_iter

        self.convs.append(nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, stride=1, padding=0, bias=True))

        for i in range(1,n_iter):

          self.convs.append(nn.Conv2d(in_channels*(2*i), (2*(i+1))*in_channels, kernel_size=3, stride=1, padding=0, bias=True))

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.l1 = nn.Linear(486,100 )
        self.l2 = nn.Linear(100,1 )



        self.relu= nn.ReLU()
        self.Softmax= nn.Softmax(dim=1)
        self.sigmoid= nn.Sigmoid()

        self.Tanh=nn.Tanh()

        if initw:

          #xavier_normal
          nn.init.xavier_normal_(self.l1.weight)
          nn.init.xavier_normal_(self.l2.weight)
          nn.init.xavier_normal_(self.l3.weight)

    def forward(self, x):

      for i in range(len(self.convs) -1):
            x = self.convs[i](x)
            x = torch.relu(x)
            x = self.maxpool(x)             

      x = self.convs[-1](x)

      x = x.view(x.shape[0], -1)

      x = self.relu(self.l1(x))
      x = self.l2(x)

      return x
    

class EncoderUNET(nn.Module):
    def __init__(self, in_channels, initw = False):
        super().__init__()

        self.unet = unet.build_unet(in_channels)

        self.l1 = nn.Linear(2500,500 )
        self.l2 = nn.Linear(500,100 )
        self.l3 = nn.Linear(100,1 )



        self.relu= nn.ReLU()
        self.Softmax= nn.Softmax(dim=1)
        self.sigmoid= nn.Sigmoid()

        self.Tanh=nn.Tanh()

        if initw:

          #xavier_normal
          nn.init.xavier_normal_(self.l1.weight)
          nn.init.xavier_normal_(self.l2.weight)
          nn.init.xavier_normal_(self.l3.weight)

    def forward(self, x):

      x_masks = self.unet(x)

      x = x_masks[:,0:1,:,:]*x

      x = x.view(x.shape[0], -1)
      x = self.relu(self.l1(x))
      x = self.relu(self.l2(x))
      x = self.l3(x)

      return x
    
