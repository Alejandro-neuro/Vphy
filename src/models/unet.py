import torch 
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
    def forward(self, inputs, skip):
        x = self.up(inputs)

        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        
        
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x
    

class build_unet(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(in_c, in_c*2)
        self.e2 = encoder_block(in_c*2, in_c*4)
        self.e3 = encoder_block(in_c*4, in_c*8)
        """ Bottleneck """
        self.b = conv_block(in_c*8, in_c*16)
        """ Decoder """
        self.d1 = decoder_block(in_c*16, in_c*8)
        self.d2 = decoder_block(in_c*16, in_c*8)
        self.d3 = decoder_block(in_c*8, in_c*4)
        self.d4 = decoder_block(in_c*4, in_c*2)
        """ Classifier """
        self.outputs = nn.Conv2d(in_c*2, in_c, kernel_size=1, padding=0)

        self.relu = nn.ReLU()
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        """ Bottleneck """
        b = self.b(p3)
        """ Decoder """
        d2 = self.d2(b, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        return self.relu(outputs)