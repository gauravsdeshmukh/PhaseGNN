# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 17:56:25 2023

@author: Gaurav Deshmukh
"""
import argparse
import numpy as np
import torch

from phase_utils import model_training, save_results

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(prog="Train PhaseGNN", description="Train a Phase GNN model for a specified crtical property and number of epochs")
    parser.add_argument("-c","--critical-property",action="store")
    parser.add_argument("-e","--epochs",action="store")
    args = parser.parse_args()
    
    if args.epochs is None:
        epochs = 1000
    else:
        epochs = int(args.epochs)
    if args.critical_property is None:
        raise Exception("Enter critical property (either P_c or T_c)")
    else:
        critical_property = str(args.critical_property)
    
    # Set seed
    np.random.seed(1)
    torch.manual_seed(1)
    
    # Use GPU if present
    use_GPU = True if torch.cuda.is_available() else False
    
    # Make model configuration dictionary 
    config={"critical_property": critical_property, "node_vec_len": 40, "max_atoms": 100, "node_fea_len": 40, "hidden_fea_len": 40, "n_conv":4, "n_hidden": 2, "n_outputs": 1, "train_size": 0.7, "val_size": 0.2, "batch_size": 64, "use_GPU": use_GPU, "optimizer": "adam","epochs": epochs, "name": "{0}-fea40-conv4-hidden-dropout10".format(critical_property), "p_dropout": 0.1}
    
    # Perform model training
    train_losses,val_losses, train_errors, val_errors, output_true, output_pred, output_smiles=model_training(config)
    
    # Save results
    save_results(train_losses, val_losses, train_errors, val_errors, output_true, output_pred, output_smiles, config)