# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 17:56:25 2023

@author: Gaurav Deshmukh
"""
import numpy as np
import torch

from phase_utils import model_training, save_results

if __name__ == "__main__":
    # Set seed
    np.random.seed(1)
    torch.manual_seed(1)
    
    # Use GPU if present
    use_GPU = True if torch.cuda.is_available() else False
    
    # Make model configuration dictionary 
    config={"critical_property": "P_c", "node_vec_len": 40, "max_atoms": 100, "node_fea_len": 40, "hidden_fea_len": 40, "n_conv":4, "n_hidden": 2, "n_outputs": 1, "train_size": 0.7, "val_size": 0.2, "batch_size": 64, "use_GPU": use_GPU, "optimizer": "adam","epochs": 1000, "name": "Pc_fea40_conv4_hidden_dropout10", "p_dropout": 0.1}
    
    # Perform model training
    train_losses,val_losses, train_errors, val_errors, output_true, output_pred, output_smiles=model_training(config)
    
    # Save results
    save_results(train_losses, val_losses, train_errors, val_errors, output_true, output_pred, output_smiles, config)