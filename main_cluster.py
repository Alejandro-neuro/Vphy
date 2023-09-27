
import numpy as np
from omegaconf import OmegaConf
from src import models
from src import modelConv2d
from src import modelineal
from src import loss_func
from src import train
from src import loader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src import optimizer_Factory as of
import Data.genData as genData
from src import custom_plots as cp
from src import Visual_utils as vu



def main():
    g=9.81
    L=1
    x0=np.pi/2
    x1=np.pi/2.5
    t,a = genData.generatePendulumA(g,L,x0, x1)

    train_dataloader, test_dataloader, train_x, val_x  = loader.getLoader(a, type = "Dataset_decoder")
    visual_loader= loader.getLoader(a, split=False, type = "Dataset_decoder")

    linearDecoder = modelineal.Decoder(initw=True)


    linearDecoder, train_losses, val_losses, accuracy_list  = train.train(linearDecoder, train_dataloader, test_dataloader, 'linearDecoder', loss_name='Focal_batch_loss')
    vu.visualize_dec(linearDecoder, visual_loader, video_name = 'LinearDecoder_ExpVsPred_focalLoss.mp4')

if __name__ == "__main__":
    main()  