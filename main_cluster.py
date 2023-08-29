
import numpy as np
from omegaconf import OmegaConf
import models
import modelConv2d
import modelineal
import loss_func
import train
import loader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import optimizer_Factory as of
import Data.genData as genData
import custom_plots as cp
import Visual_utils as vu



def main():
    g=9.81
    L=1
    x0=np.pi/2
    x1=np.pi/2.5
    t,a = genData.generatePendulumA(g,L,x0, x1)

    train_dataloader, test_dataloader, train_x, val_x  = loader.getLoader(a, type = "Dataset_decoder")
    visual_loader= loader.getLoader(a, split=False, type = "Dataset_decoder")

    linearDecoder = modelineal.Decoder(initw=True)


    linearDecoder, train_losses, val_losses, accuracy_list  = train.train(linearDecoder, train_dataloader, test_dataloader, 'linearDecoder', loss_name='decoder_loss')
    vu.visualize_dec(linearDecoder, visual_loader, video_name = 'LinearDecoder_ExpVsPred.mp4')

if __name__ == "__main__":
    main()  