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

def latent_loss(input_img, outputs, expected_pred,print_loss = False):

    z2_encoder, z2_phys, zroll = outputs

    #print("z2_encoder",z2_encoder.shape)

    
    
    z2_encoder = z2_encoder.reshape(-1, z2_encoder.shape[-1])
    z2_phys = z2_phys.reshape(-1, z2_phys.shape[-1])  
    zroll = zroll.reshape(-1, zroll.shape[-1])
    #print("z2_encoder",z2_encoder.shape)
    loss_MSE = nn.MSELoss()
    loss1 = loss_MSE(z2_phys, z2_encoder)
    loss2 = loss_MSE(z2_encoder, zroll)
    loss =  loss2# + loss2

    

   
    KLD_loss = KL_divergence( z2_encoder)


    total_loss = loss + KLD_loss

    # if torch.isnan(loss):
    #     return KLD_loss
    # if torch.isnan(loss):
    #     return KLD_loss
    #     print("loss",loss)
    #     raise ValueError("Loss is NaN")    
    return total_loss

def KL_divergence(z):

    mu = z.mean(0)
    logvar = torch.log(z.var(0))

    KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return KLD_loss

def KL_divergence_2(z, mu_2 = 1.0, var_2 = 1.0):

    mu = z.mean(0)
    logvar = torch.log(z.var(0))
    return 0.5 * torch.sum( ((mu-mu_2).pow(2))/var_2 + logvar.exp()/var_2 - 1 - logvar - np.log(var_2) )

def latent_loss_multiple(input_img, outputs, expected_pred,print_loss = False):

    z2_encoder, z2_phys, z_renorm = outputs

    z2_encoder = z2_encoder.reshape(-1, z2_encoder.shape[2])
    z2_phys = z2_phys.reshape(-1, z2_phys.shape[2])  
    z_renorm = z_renorm.reshape(-1, z_renorm.shape[2])
    
   
    loss_MSE = nn.MSELoss()
    loss1 = loss_MSE(z2_phys, z_renorm) 
    loss2 = loss_MSE(z2_phys, z2_encoder) 
    loss3 = loss_MSE(z2_encoder, z_renorm)
    loss =  loss2+loss3

    
    
    sum_kld = 0
    # for i in range( z2_encoder.shape[-1]):

    #     if z2_encoder[:,i].mean(0) < 1e-5:
    #         ztemp = z2_encoder[:,i]
    #         sum_kld += KL_divergence(ztemp )
    #     else:
    #         ztemp = z2_encoder[:,i]/z2_encoder[:,i].mean(0)

    #         sum_kld += KL_divergence(ztemp )

    KLD_4d= KL_divergence( z2_encoder )
    KLD_2d = KL_divergence_2( z2_encoder.reshape(-1,2) )
    KLD_1d = KL_divergence_2( z2_encoder.reshape(-1) )

    KLD_loss2= KL_divergence( z2_phys )
    sum_kld2 = KL_divergence_2( z2_phys.reshape(-1) )

    # sothng happend when kld 0 and sum = 1

    KLD_loss = KLD_4d + 0*KLD_2d + 0*KLD_1d #+ KLD_loss2 + sum_kld2

    total_loss = 4*loss + KLD_loss
    if print_loss:
        print("loss",loss)
        print("KLD_4d",KLD_4d)
        print("KLD_2d",KLD_2d)
        print("KLD_1d",KLD_1d)

    # mu_2 = 32
    # var_2 = 32   
    # KLD = 0.5 * torch.sum( ((mu-mu_2).pow(2))/var_2 + logvar.exp()/var_2 - 1 - logvar - np.log(var_2) )

    if torch.isnan(loss):
        return KLD_loss
    if torch.isnan(KLD_loss):
        return loss
     
    return total_loss

def latent_loss_multiple_Kld(input_img, outputs, expected_pred):

    z2_encoder, z2_phys = outputs

    z2_encoder = z2_encoder.reshape(-1, 2)
    z2_phys = z2_phys.reshape(-1, 2)

    mu = z2_encoder.mean(0)
    logvar = torch.log(z2_encoder.var(0))



    KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 

    total_loss = KLD_loss
    return total_loss





def latent_loss_MSE(input_img, outputs, expected_pred):

    z2_encoder, z2_phys = outputs

    z2_encoder = z2_encoder.reshape(-1,4)
    z2_phys = z2_phys.reshape(-1,4)

    loss_MSE = nn.MSELoss()
    loss = loss_MSE(z2_encoder, z2_phys)     
 
    return loss

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
    if loss == "latent_loss_multiple":
        return latent_loss_multiple
    if loss == "latent_loss_MSE":
        return latent_loss_MSE
    if loss == "latent_loss_multiple_Kld":
        return latent_loss_multiple_Kld
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