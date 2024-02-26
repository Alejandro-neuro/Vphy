
import torch
import torch.nn as nn


class mlp(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.l1 = nn.Linear(1,10 , bias=False)
        self.l2 = nn.Linear(10,1000, bias=True)
        self.l3 = nn.Linear(1000,2500, bias=False)

        self.uflat = nn.Unflatten(1, torch.Size([50,50]))

        #normalization layer
        self.norm = nn.BatchNorm1d(1000)


        #activation functions
        self.relu= nn.ReLU()
        self.Softmax= nn.Softmax(dim=1)
        self.sigmoid= nn.Sigmoid()
        self.Tanh=nn.Tanh()

        if initw:

          #uniform_
          #xavier_normal
          nn.init.xavier_normal_(self.l1.weight)
          nn.init.xavier_normal_(self.l2.weight)
          #nn.init.xavier_normal_(self.l3.weight)

    def forward(self, x):

      x = self.Tanh(self.l1(x))
      x = self.Tanh(self.l2(x))
      # add normalization layer

      x = self.norm(x)

      x = self.l3(x)
      x = self.relu(x)

      x = self.uflat(x)
      return x
    

class mlpVec(nn.Module):
    def __init__(self, initw = False):
        super().__init__()
        self.l1 = nn.Linear(6,100 , bias=False)
        self.l2 = nn.Linear(100,1000, bias=False)
        self.l3 = nn.Linear(1000,2500, bias=False)

        self.shape = nn.Parameter(torch.rand(5))

        self.uflat = nn.Unflatten(1, torch.Size([50,50]))
        self.relu= nn.ReLU()
        self.Softmax= nn.Softmax(dim=1)
        self.sigmoid= nn.Sigmoid()

        self.Tanh=nn.Tanh()

        if initw:

          #uniform_
          #xavier_normal
          nn.init.xavier_normal_(self.l1.weight)
          nn.init.xavier_normal_(self.l2.weight)
          #nn.init.xavier_normal_(self.l3.weight)

    def forward(self, x):
      
   

      expanded_vector = self.shape.unsqueeze(0).expand(x.size(0), -1)
      
      x = torch.cat((x, expanded_vector), 1)

      x = self.relu(self.l1(x))
      x = self.relu(self.l2(x))
      x = self.l3(x)
      

      x = self.uflat(x)
      return x
    
class convDecoder(nn.Module):
    def __init__(self, initw = False):
        super(convDecoder, self).__init__()
        
       
        
        # Initial fully connected layer to upscale the scalar to a 50x50 feature map
        self.linear = nn.Linear(1, 100)
        
   
        
        # Upsampling layers
        self.upconv1 = nn.ConvTranspose2d(4, 1, kernel_size=3, stride=4, padding=0, output_padding=1)
        
        # Final convolution to get to desired image size: 50x50
        self.final_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 4, 5, 5)  # Reshape to 50x50 feature map
        
        #x = nn.functional.relu(self.conv1(x))
        
        x = nn.functional.relu(self.upconv1(x))
        #x = torch.sigmoid(self.final_conv(x))
        return x