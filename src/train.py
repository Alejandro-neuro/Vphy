import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from src import loss_func
from src import optimizer_Factory
from src import custom_plots as cp

import wandb


def train_epoch(model, loader,loss_fn, optimizer, device='cpu'):
    """
    Trains a neural network model for one epoch using the specified data loader and optimizer.
    Args:
    model (nn.Module): The neural network model to be trained.
    loader (DataLoader): The PyTorch Geometric DataLoader containing the training data.
    optimizer (torch.optim.Optimizer): The PyTorch optimizer used for training the model.
    device (str): The device used for training the model (default: 'cpu').
    Returns:
    float: The mean loss value over all the batches in the DataLoader.
    """
    model.to(device)
    model.train() # specifies that the model is in training mode
    running_loss = 0.
    total_loss = 0.
    
    #loss_fn = nn.MSELoss()
    for data in loader:

        input_Data, out_Data = data

        x0 = input_Data.to(device=device, dtype=torch.float)
        x1 = out_Data.to(device=device, dtype=torch.float)
        if False:
            max_values_dim1, _ = torch.max(x1, dim=1)
            max_values_dim2, _ = torch.max(max_values_dim1, dim=1)
            max_values_dim3, _ = torch.max(max_values_dim2, dim=1)
            max_values_dim3 = max_values_dim3.unsqueeze(1)

            print(max_values_dim3)


        # Zero gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(x0)


        # Compute the loss and its gradients
        loss  = loss_fn(x0 , outputs ,x1 )

        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    total_loss = running_loss/len(loader)
    return total_loss


#
def evaluate_epoch(model, loader,loss_fn, device='cpu'):
    with torch.no_grad():
        model.to(device)
        model.eval() # specifies that the model is in evaluation mode
        running_loss = 0.

        for data in loader:

            input_Data, out_Data = data


            x0 = input_Data.to(device=device, dtype=torch.float)
            x1 = out_Data.to(device=device, dtype=torch.float)


            outputs = model(x0)

            # Compute the loss 
            
            loss = loss_fn(x0 , outputs ,x1 )

            running_loss += loss.item()

        total_loss = running_loss/len(loader)
        
    return total_loss


def train(model, train_loader, val_loader, name, loss_name=None):

    cfg = OmegaConf.load("config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_epochs = cfg.train.epochs
    loss_fn = loss_func.getLoss(loss_name)
    optimizer = optimizer_Factory.getOptimizer(model)    


    model.to(device)
    #create vectors for the training and validation loss

    wandb.init(
            # set the wandb project where this run will be logged
            project="Vphysics-Project",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": cfg.optimize.lr,
            "architecture": "Lineal",
            "dataset": "NEURON",
            "epochs": cfg.train.epochs,
            }
        )

    train_losses = []
    val_losses = []
    accuracy_list = []
    patience = 50 # patience for early stopping

    for epoch in range(1, num_epochs+1):
        # Model training
        train_loss = train_epoch(model, train_loader, loss_fn,optimizer, device=device)
        # Model validation
        val_loss = evaluate_epoch(model, val_loader, loss_fn, device=device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        wandb.log({"train_loss": train_loss, "validation_loss": val_loss,
                    "alpha": model.pModel.alpha[0].detach().cpu().numpy(), "beta": model.pModel.beta[0].detach().cpu().numpy()})
        
        if epoch % 2 == 0:
            for name, param in model.named_parameters():
                param.requires_grad = True
        if epoch % 2 == 1 or epoch > num_epochs /2:
            for name, param in model.named_parameters():
                if name != 'pModel.alpha' and name != 'pModel.beta' :  # Replace 'fc1.weight' with the parameter you want to keep trainable
                    param.requires_grad = False

        # Early stopping
        try:
            if val_losses[-1]>=val_losses[-2]:
                early_stop += 1
            if early_stop == patience:
                print("Early stopping! Epoch:", epoch)
                break
            else:
                early_stop = 0
        except:
            early_stop = 0

        if epoch%(num_epochs /10 )== 0:
            print("epoch:",epoch, "\t training loss:", train_loss,
                  "\t validation loss:",val_loss)
    wandb.finish()        
    X = []
    X.append( { 'x': range(1, num_epochs+1), 'y': train_losses, 'label': 'train_loss'} )
    X.append({'x': range(1, num_epochs+1), 'y': val_losses, 'label': 'val_loss'} )
    cp.plotMultiple( X,  'epoch', 'Loss','Performance', name, styleDark = True )
    return model, train_losses, val_losses, accuracy_list  

if __name__ == '__main__':
    train()