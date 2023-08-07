import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
import loss_func
import optimizer_Factory
import custom_plots as cp


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

        inputs = data
        labels = data['y'].unsqueeze(1)

        inputs = inputs.to(device=device)
        labels = labels.to(device = device)
        # Zero gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)


        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
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
        accuracy=0.
        correct = 0.


        for data in loader:


            inputs = data
            labels = data['y'].unsqueeze(1)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Make predictions for this batch
            outputs = model(inputs)

            running_loss += loss_fn(outputs, labels).item()

            pred = outputs.argmax(dim=1)
            correct += (pred == labels.argmax(dim=-1) ).sum().float() / pred.shape[0] # Check against ground-truth labels.
        loss = running_loss/len(loader)
        accuracy = correct.item() / len(loader)
        
    return loss, accuracy


def train(model, train_loader, val_loader, name):

    cfg = OmegaConf.load("config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_epochs = cfg.train.epochs
    loss_fn = loss_func.getLoss()
    optimizer = optimizer_Factory.getOptimizer(model)    


    model.to(device)
    #create vectors for the training and validation loss

    train_losses = []
    val_losses = []
    accuracy_list = []
    patience = 15 # patience for early stopping

    for epoch in range(1, num_epochs+1):
        # Model training
        train_loss = train_epoch(model, train_loader, loss_fn,optimizer, device=device)
        # Model validation
        val_loss,accuracy = evaluate_epoch(model, val_loader, loss_fn, device=device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracy_list.append(accuracy)
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
                  "\t validation loss:",val_loss, 
                  "\t accuracy :", accuracy )
            
    X = []
    X.append( { 'x': range(1, num_epochs+1), 'y': train_losses, 'label': 'train_loss'} )
    X.append({'x': range(1, num_epochs+1), 'y': val_losses, 'label': 'val_loss'} )
    cp.plotMultiple( X,  'epoch', 'Loss','Performance', name, styleDark = True )
    return model, train_losses, val_losses, accuracy_list  

if __name__ == '__main__':
    train()