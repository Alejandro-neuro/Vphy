import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, initw = False):
        super().__init__()

        #Conv layers
        self.Conv = nn.Conv2d(16, 33, 3, stride=2)

        #Linear layers
        self.l1 = nn.Linear(101,50 )
        self.l2 = nn.Linear(50,10 )
        self.l3 = nn.Linear(10,1 )
        
        #Activation functions
        self.sigmoid = nn.Sigmoid()
        self.Tanh=nn.Tanh()
        self.relu= nn.ReLU()

        if initw:
          nn.init.xavier_normal(self.l1.weight)
          nn.init.xavier_normal(self.l2.weight)
          nn.init.xavier_normal(self.l3.weight)

    def forward(self, x):
      x=torch.flatten(x,1)
      x = self.Tanh(self.l1(x))
      x = self.Tanh(self.l2(x))
      x = self.l3(x)

      return x

class pModel(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.alpha = torch.tensor([-2.00], requires_grad=True).float()
        self.beta = torch.tensor([-2500.00], requires_grad=True).float()
        #self.alpha = nn.Parameter(self.alpha )
        #self.beta= nn.Parameter(self.beta)

    def forward(self, x0,x1,dt):
      


      return x1+(x1-x0)+self.alpha*(x1-x0 )*dt*2 + (self.beta*x1 )*dt*dt*4

class Decoder(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.l1 = nn.Linear(1,10 )
        self.l2 = nn.Linear(10,50 )
        self.l3 = nn.Linear(50,101 )

        self.uflat = nn.Unflatten(1, torch.Size([101,1]))
        self.relu= nn.ReLU()
        self.Softmax= nn.Softmax(dim=1)
        self.sigmoid= nn.Sigmoid()

        self.Tanh=nn.Tanh()

        if initw:

          #uniform_
          #xavier_normal
          nn.init.xavier_normal(self.l1.weight)
          nn.init.xavier_normal(self.l2.weight)
          nn.init.xavier_normal(self.l3.weight)

    def forward(self, x):

      x = self.Tanh(self.l1(x))
      x = self.Tanh(self.l2(x))
      x = self.sigmoid(self.l3(x))

      x = self.uflat(x)
      return x



class AE(nn.Module):
    def __init__(self, dt, initw = False):
        super().__init__()

        self.encoder     = Encoder(initw=initw )
        self.decoder     = Decoder(initw=initw)
        self.pModel     =  pModel(initw=initw)
        self.dt = dt

    def forward(self, x0,x1):

      z0 =self.encoder(x0)
      z1 =self.encoder(x1)
      z2 =self.pModel(z0,z1, self.dt)

      in0Rec =self.decoder(z0)
      in1Rec =self.decoder(z1)
      outRec =self.decoder(z2)

      return (z0,z1,z2), (in0Rec,in1Rec,outRec)