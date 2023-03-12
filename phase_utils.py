# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:57:20 2023

@author: Gaurav Deshmukh
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import cirpy
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors
from ctypes import ArgumentError

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from phase_gnn import PhaseGNN
from datasets.phase_data import PhaseDataset, PhaseMolecule, MolGraph, get_dataloaders, search_in_dataset

from sklearn.preprocessing import StandardScaler as SS
from sklearn.decomposition import PCA

#### CLASSES
class Standardizer:
    def __init__(self, X):
        """
        Class to standardize PhaseGNN outputs

        Parameters
        ----------
        X : torch.Tensor 
            Tensor of outputs
        """
        self.mean = torch.mean(X)
        self.std = torch.std(X)
    
    def standardize(self, X):
        """
        Convert a non-standardized output to a standardized output

        Parameters
        ----------
        X : torch.Tensor 
            Tensor of non-standardized outputs

        Returns
        -------
        Z : torch.Tensor
            Tensor of standardized outputs

        """
        Z = (X - self.mean)/(self.std)
        return Z
    
    def restore(self, Z):
        """
        Restore a standardized output to the non-standardized output

        Parameters
        ----------
        Z : torch.Tensor
            Tensor of standardized outputs

        Returns
        -------
        X : torch.Tensor 
            Tensor of non-standardized outputs

        """
        X = self.mean + Z * self.std
        return X
    
    def state(self):
        """
        Return dictionary of the state of the Standardizer

        Returns
        -------
        dict
            Dictionary with the mean and std of the outputs

        """
        return {"mean" : self.mean, "std" : self.std}
    
    def load(self, state):
        """
        Load a dictionary containing the state of the Standardizer and assign mean and std

        Parameters
        ----------
        state : dict
            Dictionary containing mean and std 
        """
        self.mean = state["mean"]
        self.std = state["std"]       


def calculate_mae(Y_true: torch.Tensor, Y_prediction: torch.Tensor):
    """
    Calculate mean absolute error between the true outputs and predictions

    Parameters
    ----------
    Y_true : torch.Tensor
        Tensor containing true outputs
    Y_prediction : torch.Tensor
        Tensor containing predictions

    Returns
    -------
    mae : torch.Tensor
        Mean absolute error

    """
    mae = torch.mean(torch.abs(Y_true - Y_prediction))
    return mae

# Utility functions to train, validate, test model
def train_model(epoch: int, model: PhaseGNN, training_dataloader: data.DataLoader, config: dict, optimizer: torch.optim.Optimizer,  loss_fn, standardizer: Standardizer, scheduler=None):
    """
    Execute training of one epoch for the PhaseGNN model.

    Parameters
    ----------
    epoch : int
        Current epoch
    model : PhaseGNN
        PhaseGNN model object
    training_dataloader : data.DataLoader
        Training DataLoader
    config : dict
        Model configuration
    optimizer : torch.optim.Optimizer
        Model optimizer 
    loss_fn : like nn.MSELoss()
        Model loss function
    standardizer : Standardizer
        Standardizer object
    scheduler : optional
        Scheduler  for learning rate. The default is None.

    Returns
    -------
    avg_loss : float
        Training loss averaged over batches
    avg_mae : float
        Training MAE averaged over batches
    """
    
    # Create variables to store losses and error
    avg_loss = 0
    avg_mae = 0
    count = 0
    
    # Parameters
    use_GPU = config["use_GPU"]
    
    # Switch model to train mode
    model.train()
    
    # Go over each batch in the dataloader
    for i,dataset in enumerate(training_dataloader):
        # Unpack data
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        output = dataset[1]

        # Reshape inputs
        first_dim = int((torch.numel(node_mat)) / (config["max_atoms"] * config["node_vec_len"]))
        node_mat = node_mat.reshape(first_dim, config["max_atoms"], config["node_vec_len"])
        adj_mat = adj_mat.reshape(first_dim, config["max_atoms"], config["max_atoms"])

        # Standardize output
        output_std = standardizer.standardize(output)
        
        # Convert tensors to Variable 
        # Check if GPU is enabled
        if use_GPU:
            nn_input = (Variable(node_mat.cuda()), Variable(adj_mat.cuda()) )
            nn_output = Variable(output_std.cuda())
        else:
            nn_input = (Variable(node_mat), Variable(adj_mat))
            nn_output = Variable(output)
            
        # Compute output from network
        nn_prediction = model(*nn_input)
        
        # Calculate loss
        loss = loss_fn(nn_output, nn_prediction)
        avg_loss += loss
        
        # Calculate MAE
        prediction = standardizer.restore(nn_prediction.cpu())
        mae = calculate_mae(output, prediction)
        avg_mae += mae
        
        # Set zero gradients for all tensors
        optimizer.zero_grad()
        
        # Do backward prop
        loss.backward()
        
        # Update optimizer parameters
        optimizer.step()
        
        # Update scheduler if present
        if scheduler is not None:
            scheduler.step()
        
        # Increase count
        count += 1
    
    # Calculate avg loss and MAE
    avg_loss = avg_loss / count
    avg_mae = avg_mae / count
        
    # Print stats 
    print("Epoch: [{0}]\tTraining Loss: [{1:.2f}]\tTraining MAE: [{2:.2f}]".format(epoch,avg_loss,avg_mae))
    
    # Return loss and MAE
    return avg_loss, avg_mae

def validate_model(epoch: int, model: PhaseGNN, validation_dataloader: data.DataLoader, config: dict, loss_fn, standardizer: Standardizer, return_predictions: bool=False):
    """
   Execute validation of one epoch for the PhaseGNN model.

   Parameters
   ----------
   epoch : int
       Current epoch
   model : PhaseGNN
       PhaseGNN model object
   vallidation_dataloader : data.DataLoader
       Validation DataLoader
   config : dict
       Model configuration
   loss_fn : like nn.MSELoss()
       Model loss function
   standardizer : Standardizer
       Standardizer object
   return_predictions: bool, optional
       Flag to return either the loss and error (False) or outputs and predictions (True). Recommended to use the former for validation and latter for testing. The default is False.

   Returns
   -------
   output: numpy.ndarray
       Array of outputs
   predictions: numpy.ndarray
       Array of predictions
   smiles: numpy.ndarray
       Array of smiles 
       
       OR
       
   avg_loss : float
       Validation loss averaged over batches
   avg_mae : float
       Validation MAE averaged over batches

    """
    
    # Create variables to store losses and error
    avg_loss = 0
    avg_mae = 0
    count = 0
    outputs = []
    predictions = []
    smiles = []
    
    # Parameters
    use_GPU = config["use_GPU"]
    
    # Switch model to train mode
    model.eval()
    
    # Go over each batch in the dataloader
    for i,dataset in enumerate(validation_dataloader):
        # Unpack data
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        output = dataset[1]
        smile = dataset[2]
        outputs.append(output.numpy())
        smiles.append(smile)

        # Reshape inputs
        first_dim = int((torch.numel(node_mat)) / (config["max_atoms"] * config["node_vec_len"]))
        node_mat = node_mat.reshape(first_dim, config["max_atoms"], config["node_vec_len"])
        adj_mat = adj_mat.reshape(first_dim, config["max_atoms"], config["max_atoms"])

        # Standardize output
        output_std = standardizer.standardize(output)
        
        # Convert tensors to Variable 
        # Check if GPU is enabled
        if use_GPU:
            with torch.no_grad():
                nn_input = (Variable(node_mat.cuda()), Variable(adj_mat.cuda()) )
                nn_output = Variable(output_std.cuda())
        else:
            with torch.no_grad():
                nn_input = (Variable(node_mat), Variable(adj_mat))
                nn_output = Variable(output)
            
        # Compute output from network
        nn_prediction = model(*nn_input)
        
        # Calculate loss
        loss = loss_fn(nn_output, nn_prediction)
        avg_loss += loss
        
        # Calculate MAE
        prediction = standardizer.restore(nn_prediction.cpu())
        predictions.append(prediction.detach().numpy())
        mae = calculate_mae(output, prediction)
        avg_mae += mae
        
        # Increase count
        count += 1
    
    # Calculate avg loss and MAE
    avg_loss = avg_loss / count
    avg_mae = avg_mae / count
        
    # Convert outputs and predictions to numpy arrays
    outputs = np.concatenate(outputs)
    predictions = np.concatenate(predictions) 
    smiles = np.concatenate(smiles)
    
    # Return loss and MAE
    if return_predictions:
        # Print stats 
        print("\tTest Loss: [{0:.2f}]\tTest MAE: [{1:.2f}]".format(avg_loss,avg_mae))
        return outputs, predictions, smiles
    else:
        # Print stats 
        print("\tValidation Loss: [{0:.2f}]\tValidation MAE: [{1:.2f}]".format(avg_loss,avg_mae))
        return avg_loss, avg_mae

def save_model(epoch: int, name: str, loss: torch.Tensor, best_status: bool, current_model: PhaseGNN, current_optimizer: torch.optim.Optimizer, current_standardizer: Standardizer, filename="checkpoint.pt"):
    """
    Save model at current epoch

    Parameters
    ----------
    epoch : int
        Current epoch
    name : str
        Name of model
    loss : torch.Tensor
        Value of loss function
    best_status : bool
        Boolean indicating whether it is the current best model (True) or not (False)
    current_model : PhaseGNN
        PhaseGNN model object
    current_optimizer : torch.optim.Optimizer
        Model optimizer
    current_standardizer : Standardizer
        Model standardizer
    filename : TYPE, optional
        Filename for model. The default is "checkpoint.pt".
    """
    
    save_dict = {"epoch": epoch, "model_state_dict": current_model.state_dict(), "optimizer_state_dict": current_optimizer.state_dict(), "standardizer_state_dict": current_standardizer.state(), }
    current_dir_path = os.path.dirname(os.path.abspath(__file__))
    torch.save(save_dict, "{0}/models/{1}/{2}".format(current_dir_path,name,filename))
    if best_status:
        torch.save(save_dict, "{0}/models/{1}/model.pt".format(current_dir_path,name))
        
def load_model(config: dict):
    """
    Load a PhaseGNN model

    Parameters
    ----------
    config : dict
        Model configuration

    Returns
    -------
    load_dict : dict
        Dictionary containing current epoch "epoch", model state dictionary "model_state_dict", standardizer state "standardizer_state_dict", optimizer state dictionary "optimizer_state_dict"
    """
    
    name = config["name"]
    current_dir_path = os.path.dirname(os.path.abspath(__file__))
    load_dict = torch.load("{0}/models/{1}/model.pt".format(current_dir_path,name))
    return load_dict

def model_training(config: dict):
    """
    Train a PhaseGNN model

    Parameters
    ----------
    config : dict
        Model configuration

    Returns
    -------
    train_losses : list
        Training losses
    val_losses : list
        Validation losses
    train_mean_errors : list
        Training MAEs
    val_mean_errors : list
        Validation MAEs
    output_true: numpy.ndarray 
        True outputs
    output_pred : numpy.ndarray
        Predicted outputs
    output_smiles : numpy.ndarray
        SMILES strings of molecules

    """
    
    # Create dataset
    data = PhaseDataset(config["critical_property"], node_vec_len=config["node_vec_len"],max_atoms=config["max_atoms"])
    data_dict = get_dataloaders(data, train_size=config["train_size"],val_size=config["val_size"],batch_size=config["batch_size"])
    
    # Print data stats
    print("Loaded PhaseDataset")
    print("Dataset size: {0}".format(len(data)))
    print("Training set size: {0}".format(data_dict["train_size"]))
    print("Validation set size: {0}".format(data_dict["val_size"]))
    print("Test set size: {0}".format(data_dict["test_size"]))
    
    # Create model
    if "p_dropout" not in config.keys():
        config["p_dropout"] = 0.
    model = PhaseGNN(node_vec_len=config["node_vec_len"], node_fea_len=config["node_fea_len"], hidden_fea_len=config["hidden_fea_len"], n_conv=config["n_conv"], n_hidden=config["n_hidden"], n_outputs=config["n_outputs"], p_dropout = config["p_dropout"])
    
    # Transfer model to GPU if applicable
    if config["use_GPU"]:
        model.cuda()
    
    # Define loss function
    loss_fn = nn.MSELoss()
    
    # Define standardizer and set mean, std based on training outputs
    train_outputs = data_dict["train_outputs"]
    standardizer = Standardizer(train_outputs)
    
    # Create optimizer 
    if config["optimizer"].lower().strip() in "adam":
        optimizer = torch.optim.Adam(model.parameters())
    elif config["optimizer"].lower().strip() in "sgd":
        optimizer = torch.optim.SGD(model.parameters())
    else:
        raise Exception("Please enter either adam or sgd as the name of the optimizer")
        
    # Create scheduler if LR milestones are given
    if "lr_milestones" in config.keys():
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config["lr_milestones"])
    else:
        scheduler = None
        
    # Create lists to store losses and MAEs
    train_losses = []
    train_mean_errors = []
    val_losses = []
    val_mean_errors = []
    
    # Create folder to store models
    curr_dir_path = os.path.dirname(os.path.abspath(__file__))
    os.makedirs("{0}/models/{1}".format(curr_dir_path,config["name"]),exist_ok=True)
        
    prev_val_loss=1e10
    
    # Iterate over epochs and train model
    for i in range(config["epochs"]):
        # Train model
        train_loss, train_error = train_model(epoch=i, model=model, training_dataloader=data_dict["train_loader"], config=config, optimizer=optimizer, loss_fn=loss_fn, standardizer=standardizer, scheduler=scheduler)
        
        # Validate model
        val_loss, val_error = validate_model(epoch=i, model=model, validation_dataloader=data_dict["val_loader"], config=config, loss_fn=loss_fn, standardizer=standardizer)
        
        # Best status
        if val_loss < prev_val_loss:
            best_status = True
            prev_val_loss = val_loss
        else:
            best_status = False
        
        # Add losses and errors to lists
        train_losses.append(train_loss.cpu().detach().numpy())
        val_losses.append(val_loss.cpu().detach().numpy())
        train_mean_errors.append(train_error.cpu().detach().numpy())
        val_mean_errors.append(val_error.cpu().detach().numpy())
        
        # Save model
        save_model(epoch=i, name=config["name"], loss=val_loss, best_status=best_status, current_model=model, current_optimizer=optimizer, current_standardizer=standardizer)
        
    # Test model
    best_model_dict = load_model(config)
    model.load_state_dict(best_model_dict["model_state_dict"])
    standardizer.load(best_model_dict["standardizer_state_dict"])
    
    output_true, output_pred, output_smiles = validate_model(epoch=i, model=model, validation_dataloader=data_dict["test_loader"], config=config, loss_fn=loss_fn, standardizer=standardizer, return_predictions=True)
    
    # Return losses
    return train_losses, val_losses, train_mean_errors, val_mean_errors, output_true, output_pred, output_smiles

def evaluate_molecule(molecule_identifier: str, config: dict):
    """
    Evaluate the critical property of a given molecule with a PhaseGNN model

    Parameters
    ----------
    molecule_identifier : str
        Molecule identifier including common name, IUPAC name, CAS number, or SMILES
    config : dict
        Model configuration

    Returns
    -------
    prediction : numpy.ndarray
        Model prediction
    """
    
    # Convert to smiles
    smiles = cirpy.resolve(molecule_identifier,"smiles")
    
    # Create molecule dataset
    dataset = PhaseMolecule(smiles, config["critical_property"], config["node_vec_len"], config["max_atoms"])
    
    # Load model 
    model = PhaseGNN(node_vec_len=config["node_vec_len"], node_fea_len=config["node_fea_len"], hidden_fea_len=config["hidden_fea_len"], n_conv=config["n_conv"], n_hidden=config["n_hidden"], n_outputs=config["n_outputs"])
    
    # Create standardizer
    standardizer = Standardizer(torch.Tensor(np.zeros(1)))
    
    # Load state
    state = load_model(config)
    model.load_state_dict(state["model_state_dict"])
    standardizer.load(state["standardizer_state_dict"])
    
    # Evaluate output
    model.eval()
    (nn_input), smiles = dataset[0]
    prediction_std = model(*nn_input)
    prediction = standardizer.restore(prediction_std).detach().numpy()
    
    # Return prediction
    return prediction

def evaluate_molecule_and_compare(molecule_identifier: str, config: dict):
    """
    Evaluate critical property of molecule and compare with value in dataset, or if not in dataset, compare with value in NIST database

    Parameters
    ----------
    molecule_identifier : str
        Molecule identifier including common name, IUPAC name, CAS number, or SMILES
    config : dict
        Model configuration
    """
    
    # Call evaluate_molecule function
    prediction = evaluate_molecule(molecule_identifier, config)
    
    # Search in dataset
    bool_dataset, value = search_in_dataset(molecule_identifier, config["critical_property"])
    
    # Print
    if bool_dataset:
        print("In dataset")
    else:
        print("Not in dataset")
    
    print("True value: {0}".format(value))
    print("Predicted value: {0}".format(prediction.flatten()[0]))

def get_embeddings(config: dict):
    """
    Return PhaseGNN embeddings

    Parameters
    ----------
    config : dict
        Model configuration

    Returns
    -------
    embeddings : numpy.ndarray
        Pooled feature vector for each molecule
    smiles : TYPE
        SMILES string for each molecule
    """
    
    # Create dataset
    data = PhaseDataset(config["critical_property"], node_vec_len=config["node_vec_len"],max_atoms=config["max_atoms"])
    data_dict = get_dataloaders(data, train_size=config["train_size"],val_size=config["val_size"],batch_size=config["batch_size"])
    total_loader = data_dict["total_loader"]
    
    
    # Create model
    if "p_dropout" not in config.keys():
        config["p_dropout"] = 0.
    model = PhaseGNN(node_vec_len=config["node_vec_len"], node_fea_len=config["node_fea_len"], hidden_fea_len=config["hidden_fea_len"], n_conv=config["n_conv"], n_hidden=config["n_hidden"], n_outputs=config["n_outputs"], p_dropout = config["p_dropout"])
    
    # Get state dict for best model 
    best_dict = load_model(config)
    model.load_state_dict(best_dict["model_state_dict"])
    
    # Check GPU use
    use_GPU = config["use_GPU"]
    if use_GPU:
        model.cuda()
    
    # Set eval mode
    model.eval()
    
    # Make empty list
    list_of_pooled_feas = []
    list_of_smiles = []
    
    # Go over every point and return pooled features
    for i,dataset in enumerate(total_loader):
        # Unpack
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        smile = dataset[2]
        list_of_smiles.append(smile)
        
        # Reshape inputs
        first_dim = int((torch.numel(node_mat)) / (config["max_atoms"] * config["node_vec_len"]))
        node_mat = node_mat.reshape(first_dim, config["max_atoms"], config["node_vec_len"])
        adj_mat = adj_mat.reshape(first_dim, config["max_atoms"], config["max_atoms"])
        
        # Check if GPU is enabled
        if use_GPU:
            with torch.no_grad():
                nn_input = (Variable(node_mat.cuda()), Variable(adj_mat.cuda()) )
        else:
            with torch.no_grad():
                nn_input = (Variable(node_mat), Variable(adj_mat))
        
        # Get pooled features
        pooled_fea = model.return_pooled_features(*nn_input)
        list_of_pooled_feas.append(pooled_fea.cpu().detach().numpy())
        
    # Reshape outputs
    smiles = np.concatenate(list_of_smiles)
    embeddings = np.concatenate(list_of_pooled_feas).reshape((len(data),config["node_fea_len"]))
    
    return embeddings,smiles
    
    

def make_loss_plot(train_losses: list, val_losses: list, train_errors: list, val_errors: list, config: dict):
    """
    Make plot of losses and MAEs against epochs.

    Parameters
    ----------
    train_losses : list
        List of training losses
    val_losses : list
        List of validation losses
    train_errors : list
        List of training errors
    val_errors : list
        List of validation errors
    config : dict
        Model configuration
    """
    
    # Unpack dict
    epochs = np.arange(0,config["epochs"],1)
    name = config["name"]   
    
    # Make figure
    fig,ax=plt.subplots(2,1,figsize=(8,7),dpi=500)
    
    # Plot training losses
    ax[0].plot(epochs,train_losses,label="Train",color="firebrick")
    
    # Plot validation losses
    ax[0].plot(epochs,val_losses,label="Val",color="royalblue")
    
    # Label
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    
    # Plot training errors
    ax[1].plot(epochs,train_errors,label="Train",color="firebrick")
    
    # Plot validation errors
    ax[1].plot(epochs,val_errors,label="Val",color="royalblue")
    
    # Set labels
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("MAE")
    
    # Make legend 
    ax[0].legend()
    ax[1].legend()
    
    # Tight layout
    fig.tight_layout()
    
    # Save figure
    curr_dir_path=os.path.dirname(os.path.abspath(__file__))
    plot_path=os.path.join(curr_dir_path,"models",name,"loss_plot.png")
    fig.savefig(plot_path)
    
def make_parity_plot(output_true: np.ndarray, output_pred: np.ndarray, config: dict):
    """
    Make a parity plot of true outputs against model predictions/

    Parameters
    ----------
    output_true : np.ndarray
        True outputs in test set
    output_pred : np.ndarray
        Model predictions in test set
    config : dict
        Model configuration
    """
    
    # Unpack dict
    name = config["name"]
    
    # Flatten outputs (only works if there is one input)
    Y_true = output_true.flatten()
    Y_pred = output_pred.flatten()
    
    # Make figure
    fig,ax=plt.subplots(1,1,figsize=(5,5),dpi=500)
    
    # Scatter plot
    ax.scatter(Y_true, Y_pred, color="royalblue", marker="o")
    
    # Plot line
    min_scatter = np.amin(np.concatenate((Y_true,Y_pred)))
    max_scatter = np.amax(np.concatenate((Y_true,Y_pred)))
    ax.plot([min_scatter*0.96,max_scatter*1.04],[min_scatter*0.96, max_scatter*1.04], color="black")
    ax.margins(x=0,y=0)
    
    # Mark error
    rmse = np.sqrt(np.mean((Y_true-Y_pred)**2))
    mae = np.mean(np.abs(Y_true-Y_pred))
    avg_scatter_x = np.mean(Y_pred)
    avg_scatter_y = np.mean(Y_true)
    ax.text(avg_scatter_x, avg_scatter_y*0.75, "MAE={0:.2f}\nRMSE={1:.2f}".format(mae,rmse))
    
    # Set limits
    ax.set_xlim([min_scatter*0.96, max_scatter*0.96])
    ax.set_ylim([min_scatter*0.96, max_scatter*0.96])
    
    # Save plot
    curr_dir_path=os.path.dirname(os.path.abspath(__file__))
    plot_path=os.path.join(curr_dir_path,"models",name,"parity_plot.png")
    fig.savefig(plot_path)

def save_test_csv(output_true: np.ndarray, output_pred: np.ndarray, output_smiles: np.ndarray, config: dict):
    """
    Save a csv file containing the true outputs, model predicitons, and smiles strings of molecules.

    Parameters
    ----------
    output_true : np.ndarray
        True outputs in the test set
    output_pred : np.ndarray
        Model predictions in the test set
    output_smiles : np.ndarray
        SMILES strings in the test set
    config : dict
        Model configuration
    """
    
    # Unpack dict
    name = config["name"]
    
    # Get number of outputs
    n_outputs = config["n_outputs"]
    
    # Create dataframe
    df = pd.DataFrame({"smiles":output_smiles})
    
    # Add output columns
    for i in range(n_outputs):
        df["Y_true_{0}".format(i)]=output_true[:,i]
        df["Y_pred_{0}".format(i)]=output_pred[:,0]
        
    # Save csv
    curr_dir_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(curr_dir_path,"models",name,"test.csv")
    df.to_csv(csv_path)

def save_config(config: dict):
    """
    Save model configuration as a text file.

    Parameters
    ----------
    config : dict
        Model configuration
    """
    
    name = config["name"]
    curr_dir_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(curr_dir_path,"models",name,"config.txt")
    with open(config_path,"w") as f:
        for c in config.keys():
            print("{0}:{1}".format(c,config[c]),file=f)

def save_results(train_losses: list, val_losses: list, train_errors: list, val_errors: list, output_true: np.ndarray, output_pred: np.ndarray, output_smiles: np.ndarray, config: dict):
    """
    Wrapper function to save plots, csv, and configuration

    Parameters
    ----------
    train_losses : list
        List of training losses
    val_losses : list
        List of validation losses
    train_errors : list
        List of training errors
    val_errors : list
        List of validation errors
    output_true : numpy.ndarray
        True outputs in test set
    output_pred : numpy.ndarray
        Model predictions in test set
    output_smiles : np.ndarray
        SMILES strings in test set
    config : dict
        Model configuration
    """
    
    make_loss_plot(train_losses,val_losses,train_errors,val_errors, config)
    make_parity_plot(output_true, output_pred, config)
    save_test_csv(output_true, output_pred, output_smiles, config)
    save_config(config)

if __name__ == "__main__":
    # Set seed
    np.random.seed(1)
    torch.manual_seed(1)
    
    use_GPU = True if torch.cuda.is_available() else False
    config={"critical_property": "P_c", "node_vec_len": 40, "max_atoms": 100, "node_fea_len": 40, "hidden_fea_len": 40, "n_conv":4, "n_hidden": 2, "n_outputs": 1, "train_size": 0.7, "val_size": 0.2, "batch_size": 64, "use_GPU": use_GPU, "optimizer": "adam","epochs": 1000, "name": "Pc_fea40_conv4_hidden_dropout10", "p_dropout": 0.1}
    #train_losses,val_losses, train_errors, val_errors, output_true, output_pred, output_smiles=model_training(config)
    #save_results(train_losses, val_losses, train_errors, val_errors, output_true, output_pred, output_smiles, config)
    molecule_name = "acetaldehyde"
    evaluate_molecule_and_compare(molecule_name, config)
    
    
    # # PCA
    # embeddings, smiles = get_embeddings(config)
    
    # mw = []
    # for smile in smiles:
    #     mol = MolGraph(smile,node_vec_len=40,max_atoms=100)
    #     mw.append(Descriptors.ExactMolWt(mol.mol))
        

    # # Do scaling
    # X = SS().fit_transform(embeddings)
    
    # # Do PCA
    # pca = PCA(n_components=2)
    # X_pc = pca.fit_transform(X)
    
    # # Plot
    # plt.figure(figsize=(5,5),dpi=500)
    # plt.scatter(X_pc[:,0], X_pc[:,1],c=mw)
    # plt.show()
    
    
        