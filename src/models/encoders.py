import torch
import torch.nn as nn



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