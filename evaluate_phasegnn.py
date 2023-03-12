# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 17:59:04 2023

@author: Gaurav Deshmukh
"""
import argparse
import torch
from phase_utils import evaluate_molecule, evaluate_molecule_and_compare

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(prog="Evaluate PhaseGNN", description="Evaluate a Phase GNN model for a specified crtical property and molecule name")
    parser.add_argument("-c","--critical-property",action="store")
    parser.add_argument("-n","--name",action="store",nargs="+")
    args = parser.parse_args()
    
    # Critical property
    critical_property = str(args.critical_property)
    
    # Specify molecule name
    molecule_name = str(" ".join(args.name))
    
    # Use GPU if present
    use_GPU = True if torch.cuda.is_available() else False
    
    # Make model configuration dictionary 
    config={"critical_property": critical_property, "node_vec_len": 40, "max_atoms": 100, "node_fea_len": 40, "hidden_fea_len": 40, "n_conv":4, "n_hidden": 2, "n_outputs": 1, "train_size": 0.7, "val_size": 0.2, "batch_size": 64, "use_GPU": use_GPU, "optimizer": "adam","epochs": 1000, "name": "{0}-fea40-conv4-hidden-dropout10".format(critical_property), "p_dropout": 0.1}
    
    # Evaluate critical property and print output
    evaluate_molecule_and_compare(molecule_name, config)