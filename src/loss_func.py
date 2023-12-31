import torch
import torch.nn as nn
from omegaconf import OmegaConf

def custom_loss(input_img, outputs, expected_pred):
    lossMSE = nn.MSELoss()

    x0=input_img[:,-2,:,:]
    x1=input_img[:,-1,:,:]
    z,rec=outputs
    rec0,rec1,outrec=rec
    z0,z1,z2 = z

    zed = torch.concatenate((z0,z1,z2), axis=1)

    #print("zed.shape",zed.shape)

    expected_pred=expected_pred.squeeze(1)


    l1=lossMSE(rec0, x0)
    l2=lossMSE(rec1, x1)
    l3=lossMSE(outrec,expected_pred)


    return l1+l2+l3

def Focal_batch_loss(input_img, output, expected_pred):
    lossMSE = nn.MSELoss()

   

    expected_pred=expected_pred.squeeze(1)

    diff_pred = torch.mean((output - expected_pred) ** 2,  (1,2,3))

    #diff_rec = torch.mean((rec0 - x0) ** 2,  (1,2,3))

    batch_loss = torch.mean(torch.exp(diff_pred )-1)


    return batch_loss

def decoder_loss(input_img, output, expected_pred):
    lossMSE = nn.MSELoss()
        

    expected_pred = expected_pred.squeeze(1)
    
    return lossMSE(output,expected_pred) #+ lossMSE(output.sum(1),expected_pred.sum(1)) + lossMSE(output.sum(2),expected_pred.sum(2)) + lossMSE(output.sum((1,2)).unsqueeze(1),expected_pred.sum((1,2)).unsqueeze(1))
def getLoss(loss = None):

    if loss == None:

        cfg = OmegaConf.load("config.yaml")
        loss = cfg.loss
    
    if loss == "Focal_batch_loss":
        return Focal_batch_loss
    if loss == "MSE":
        loss_fn = nn.MSELoss()
        return loss_fn
    if loss == "MAE":
        loss_fn = nn.L1Loss()
        return loss_fn 
    if loss == "custom":   
        return custom_loss  
    if loss == "BCE":
        loss_fn = nn.BCELoss()
        return loss_fn  
    if loss == "CE":
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn  
    if loss == "decoder_loss":
        loss_fn = decoder_loss
        return loss_fn  
    if loss == "NLL":
        loss_fn = nn.NLLLoss()
        return loss_fn  
    pass

if __name__ == "__main__":
    getLoss()