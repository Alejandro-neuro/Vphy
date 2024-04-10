import torch
import torch.nn as nn
import torch.nn.functional as F



class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 2, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):

        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


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


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x 


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        if x.shape[-2] != skip_x.shape[-2]:
            x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x 


class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, img_size = 50, device="cuda"):
        super().__init__()
        self.device = device
        
        self.inc = DoubleConv(c_in, c_in*2 )
        self.down1 = Down(c_in*2 , c_in*4 )
        self.sa1 = SelfAttention(c_in*4 , int(img_size/2) )
        self.down2 = Down(c_in*4, c_in*8 )
        self.sa2 = SelfAttention(c_in*8, int(img_size/4) )
        self.down3 = Down(c_in*8, c_in*8)
        self.sa3 = SelfAttention(c_in*8 , int(img_size/8) )

        self.bot1 = DoubleConv(c_in*8 , c_in*16)
        self.bot2 = DoubleConv(c_in*16, c_in*16)
        self.bot3 = DoubleConv(c_in*16, c_in*8)

        self.up1 = Up(c_in*16 , c_in*4 )
        self.sa4 = SelfAttention(c_in*4 , int(img_size/4))
        self.up2 = Up(c_in*8 , c_in*2 )
        self.sa5 = SelfAttention(c_in*2 , int(img_size/2))
        self.up3 = Up(c_in*4 , c_in*2 )
        self.sa6 = SelfAttention(c_in*2 , img_size)
        self.outc = nn.Conv2d(c_in*2 , c_out, kernel_size=1)


    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.sa1(x2)
        x3 = self.down2(x2)
        x3 = self.sa2(x3)
        x4 = self.down3(x3)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3)
        #print("x.shape sa4",x.shape)
        x = self.sa4(x)
        x = self.up2(x, x2)
        #print("x.shape sa5",x.shape)
        x = self.sa5(x)
        x = self.up3(x, x1)
        x = self.sa6(x)
        output = self.outc(x)
        return output