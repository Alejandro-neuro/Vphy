import torch
import torch.nn as nn
from omegaconf import OmegaConf
import numpy as np

def getMaxValues(x1):
    max_values_dim1, _ = torch.max(x1, dim=1)
    max_values_dim2, _ = torch.max(max_values_dim1, dim=1)
    max_values_dim3, _ = torch.max(max_values_dim2, dim=1)
    max_values_dim3 = max_values_dim3.unsqueeze(1)

    return max_values_dim3

def MSEVARLoss(pred_img, expected_pred):

    lossMSE = nn.MSELoss()

    pred_img_var = pred_img.view(pred_img.shape[0], -1)
    pred_img_var = torch.var(pred_img_var, dim=0)
    pred_img_var = torch.mean(pred_img_var)
    
    expected_pred_var = expected_pred.view(expected_pred.shape[0], -1)
    expected_pred_var = torch.var(expected_pred_var, dim=0)
    expected_pred_var = torch.mean(expected_pred_var)

    return lossMSE(pred_img, expected_pred) + lossMSE(pred_img_var, expected_pred_var)

def manual_MSE(pred_img, expected_pred):

    squared_errors = (pred_img - expected_pred) ** 2
    log_errors = 10*torch.log(squared_errors +1)
    return torch.mean(log_errors)

def custom_loss(input_img, outputs, expected_pred):
    lossMSE = nn.MSELoss()

    x0=input_img[:,-2,:,:]
    x1=input_img[:,-1,:,:]
    z,rec=outputs
    rec0,rec1,outrec=rec

    if False:
        temp1 = getMaxValues(x0.unsqueeze(1))
        temp2 =getMaxValues(x1.unsqueeze(1))
        temp3 =getMaxValues(expected_pred)

        temp = torch.cat((temp1,temp2,temp3),1)
        print("temp",temp)  
    z0,z1,z2 = z

    zed = torch.concatenate((z0,z1,z2), axis=1)

    #print("zed.shape",zed.shape)

    expected_pred=expected_pred.squeeze(1)

    reconstruction = torch.cat((rec0,rec1,outrec), axis=0)
    ground_truth = torch.cat((x0,x1,expected_pred), axis=0)


    return lossMSE(reconstruction,ground_truth) 

def Focal_batch_loss(input_img, output, expected_pred):
    lossMSE = nn.MSELoss()

   

    expected_pred=expected_pred.squeeze(1)

    diff_pred = torch.mean((output - expected_pred) ** 2,  (1,2,3))

    #diff_rec = torch.mean((rec0 - x0) ** 2,  (1,2,3))

    batch_loss = torch.mean(torch.exp(diff_pred )-1)


    return batch_loss

def decoder_loss(input_img, output, expected_pred):

    x1,x2 = output
    lossMSE = manual_MSE
        

    expected_pred = expected_pred.squeeze(1)
    
    return lossMSE(x1,expected_pred) + lossMSE(x2,expected_pred)
#+ lossMSE(output.sum(1),expected_pred.sum(1)) + lossMSE(output.sum(2),expected_pred.sum(2)) + lossMSE(output.sum((1,2)).unsqueeze(1),expected_pred.sum((1,2)).unsqueeze(1))
def adversarial_loss(discriminated, real):
    loss = nn.BCELoss()
    return loss(discriminated, real)

def latent_loss(input_img, outputs, expected_pred):

    z2_encoder, z2_phys = outputs

    #print("z2_encoder",z2_encoder.shape)
    #z2_encoder = z2_encoder.reshape(-1, z2_encoder.shape[2])
    #z2_phys = z2_phys.reshape(-1, z2_phys.shape[2])
    

    #print("z2_encoder",z2_encoder.shape)
    #print("z2_phys",z2_phys.shape)
    loss_MSE = nn.MSELoss()
    loss = loss_MSE(z2_encoder, z2_phys)

    
    #z2_encoder = z2_encoder.reshape(-1,2)
    #z2_phys = z2_phys.reshape(-1, 2)

    z2_encoder = z2_encoder.reshape(-1, 2)
    z2_phys = z2_phys.reshape(-1, 2)
    
    mu = z2_encoder.mean(0)
    logvar = torch.log(z2_encoder.var(0))

    #mu = mu.view(-1)
    #logvar = logvar.view(-1)

    mu_2 = 2
    var_2 = 1   
    #KLD = 0.5 * torch.sum( ((mu-mu_2).pow(2))/var_2 + logvar.exp()/var_2 - 1 - logvar - np.log(var_2) )

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # max = torch.max(z2_encoder,0)[0]
    # min = torch.min(z2_encoder,0)[0]

    # log_uniform = torch.log( (1)/(max-min) )
    # log_uniform = log_uniform.sum()
    #KLD = log_uniform

  

    total_loss = loss +KLD

    if torch.isnan(total_loss):
        print("loss",loss)
        raise ValueError("Loss is NaN")    
    return total_loss

def getLoss(loss = None):

    if loss == None:

        cfg = OmegaConf.load("config.yaml")
        loss = cfg.loss
    
    if loss == "Focal_batch_loss":
        return Focal_batch_loss
    if loss == "adversarial_loss":
        return adversarial_loss
    if loss == "MSE":
        return nn.MSELoss()
    if loss == "latent_loss":
        return latent_loss
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