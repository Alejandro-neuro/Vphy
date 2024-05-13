import numpy as np
from omegaconf import OmegaConf
from src.models import models
from src.models import model as mainmodel
from src.models import modelConv2d
from src.models import modelineal
from src.models import decoders
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
import torchvision
import wandb
import random
import os



torch.cuda.empty_cache() 
torch.manual_seed(42)


def main():


    data_folder = np.load('Data/color_mnist.npz')
    data_train = data_folder['train_x']
    #
    #data_train = new_data

    #data_train = data_merge_mask
    dt = 0.3
    train_dataloader, test_dataloader, train_x, val_x  = loader.getLoader_folder(data_train, split=True)
    latentEncoder = mainmodel.EndPhysMultiple(dt = dt, 
                                            in_size=64, 
                                            n_mask = 2, 
                                            in_channels = 2, 
                                            latent_dim=2, 
                                            pmodel = "ODE_2ObjectsSpring", 
                                            init_phys = [2.0,0.1], initw=True)
    latentEncoder, log  = train.train(latentEncoder, 
                                                train_dataloader, 
                                                test_dataloader,                                 
                                                loss_name='latent_loss')

if __name__ == "__main__":
    main()  