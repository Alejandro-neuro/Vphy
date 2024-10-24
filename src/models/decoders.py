
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import build_unet as unet


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
    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
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
        
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x 
    
class imageEnhancer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(imageEnhancer,self).__init__(*args, **kwargs)
        self.l1 = DoubleConv(1, 1)

        self.unet = unet(in_c=1)

        self.l2 = DoubleConv(1, 1)
        self.l3 = DoubleConv(1, 1)

        

        


    def forward(self, x):
        x = self.l1(x)
        unet_out = self.unet(x)
        x = unet_out * x
        x = nn.functional.relu(self.l2(x))
        x = nn.functional.relu(self.l3(x))
        return x

class convDecoder(nn.Module):
    def __init__(self, initw = False):
        super(convDecoder, self).__init__()
         
        # Initial fully connected layer to upscale the scalar to a 50x50 feature map
        self.linear = nn.Linear(1, 100)        
        # Upsampling layers
        self.upconv1 = nn.ConvTranspose2d(4, 1, kernel_size=3, stride=4, padding=0, output_padding=1)

        self.up1 = Up(4, 3)

        self.up2 = Up(3, 2)

        self.up3 = Up(2, 1)
        # Final convolution to get to desired image size: 50x50
        self.final_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        self.imageEnhancer = imageEnhancer()

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 4, 5, 5)  
        
        x = self.up1(x)
        x = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
        
        x = self.up2(x)
        x = F.pad(x, (0, 1, 0, 1), mode='constant', value=0)
        
        x = self.up3(x)
        
        #x = nn.functional.relu(self.conv1(x))
        
        x = nn.functional.relu(self.final_conv(x))

        x2 = self.imageEnhancer(x)
        
        #x = torch.sigmoid(self.final_conv(x))
        return (x,x2)
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = 50 // 4
        self.l0 = nn.Sequential(nn.Linear(1, 100))
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=2),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l0(z)
        out = self.l1(out)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 50 // 2 ** 4
        #128 * ds_size ** 2
        self.adv_layer = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
    
class convGAN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(convGAN, self).__init__(*args, **kwargs)

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
