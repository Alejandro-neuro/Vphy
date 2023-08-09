import torch
import torch.nn as nn
from omegaconf import OmegaConf

def custom_loss(input_img, outputs, expected_pred):
    lossMSE = nn.MSELoss()

    x0,x1=input_img
    z,rec=outputs
    rec0,rec1,outrec=rec
    z0,z1,z2 = z

    l1=lossMSE(rec0, x0)
    l2=lossMSE(rec1, x1)
    l3=lossMSE(outrec,expected_pred)


    return l1,l2,l3

def getLoss():
    cfg = OmegaConf.load("config.yaml")

    if cfg.loss == "MSE":
        loss_fn = nn.MSELoss()
        return loss_fn
    if cfg.loss == "MAE":
        loss_fn = nn.L1Loss()
        return loss_fn 
    if cfg.loss == "custom":   
        return custom_loss  
    if cfg.loss == "BCE":
        loss_fn = nn.BCELoss()
        return loss_fn  
    if cfg.loss == "CE":
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn  
    if cfg.loss == "NLL":
        loss_fn = nn.NLLLoss()
        return loss_fn  
    pass

if __name__ == "__main__":
    getLoss()