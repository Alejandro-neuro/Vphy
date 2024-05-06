import torch
import torch.nn as nn

from . import unet



class EncoderMLP(nn.Module):
    def __init__(self, in_size = 50,in_chan = 1,  latent_dim=1, initw = True):
        super().__init__()

        self.l1 = nn.Linear(in_size*in_size*in_chan,500*in_chan )
        self.l2 = nn.Linear(500*in_chan,100*in_chan )
        self.l3 = nn.Linear(100*in_chan,latent_dim)

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

    
      x = x.reshape(x.shape[0], -1)
      x = self.relu(self.l1(x))
      x = self.relu(self.l2(x))
      x = self.l3(x)

      return x
    
class EncoderCNN(nn.Module):
    def __init__(self, in_channels, channels = [1,32,64,128], initw = False):
        super().__init__()

        

        self.in_channels = in_channels
        self.n_iter = len(channels)
      
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(0, self.n_iter-1):
          self.convs.append(nn.Conv2d( channels[i], channels[i+1], kernel_size=3, stride=1, padding=0, bias=True))
          self.norms.append(nn.BatchNorm2d(channels[i+1]))

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.l1 = nn.Linear(56448,1000 )
        self.l2 = nn.Linear(1000,100 )
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

      for i in range(self.n_iter-2):
            x = self.convs[i](x)
            #x = self.norms[i](x)
            x = torch.relu(x)
            if i == 1:
                x = self.maxpool(x)            

      x = self.convs[-1](x)

      x = x.view(x.shape[0], -1)

      

      x = self.relu(self.l1(x))
      x = self.relu(self.l2(x))
      x = self.l3(x)

      

      return x
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,  mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class EncoderCNNnorm(nn.Module):
    def __init__(self, in_channels, n_iter, initw = False):
        super().__init__()

        self.convs = nn.ModuleList()

        self.in_channels = in_channels
        self.n_iter = n_iter

        self.convs.append(DoubleConv(in_channels, 2*in_channels))

        for i in range(1,n_iter):

          self.convs.append(DoubleConv(in_channels*(2*i), (2*(i+1))*in_channels))

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

      x_masks = self.unet(x)

      x = x_masks[:,0:1,:,:]*x

      x = x.view(x.shape[0], -1)
      x = self.relu(self.l1(x))
      x = self.relu(self.l2(x))
      x = self.l3(x)

      return x
    
