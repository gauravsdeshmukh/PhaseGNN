# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:51:02 2023

@author: Gaurav Deshmukh
"""
import os
import time
import pickle
from typing import Union

import numpy as np
import pandas as pd
import requests
import urllib
from bs4 import BeautifulSoup

import cirpy
from tqdm import tqdm
from rdkit import Chem

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class NISTData:
    def __init__(self, molecule_identifier: str):
        """
        Parameters
        ----------
        molecule_identifier : str
            Molecule identifier can be the common name, IUPAC name, CAS number or SMILES string of the molecule
        """
        
        self.identifier=molecule_identifier.strip()
        self.query_molecule_CAS()
        self.query_molecule_SMILES()
        self.query_molecule_IUPAC_name()
              
    def query_molecule_CAS(self):
        """Get the CAS number of molecule using CIRPy"""
        try:
            cas_numbers=cirpy.resolve(self.identifier,"cas")
        except (urllib.error.URLError,TimeoutError):
            cas_numbers=None
        if cas_numbers is None:
            self.cas=-1
            return
        if isinstance(cas_numbers,str):
            cas_numbers=[cas_numbers]
        filtered_cas_numbers=[]
        
        # Only keep CAS numbers that have less than 9 characters
        for n in cas_numbers:
            if len(n) <= 9:
                filtered_cas_numbers.append(n)
                
        # Set object
        self.cas=filtered_cas_numbers
               
    def query_molecule_SMILES(self):
        """Get the SMILES string of molecule using CIRPy"""
        try:
            self.smiles=cirpy.resolve(self.identifier,"smiles")
        except (urllib.error.URLError,TimeoutError):
            self.smiles=None
        if self.smiles is None:
            self.smiles=-1
    
    def query_molecule_IUPAC_name(self):
        """Get the IUPAC name(s) of molecule using CIRPy"""
        try:
            self.iupac_name=cirpy.resolve(self.identifier,"iupac_name")
        except (urllib.error.URLError,TimeoutError):
            self.iupac_name=None
        if self.iupac_name is None:
            self.iupac_name=-1
    
    def _get_phase_page(self):
        """Use requests to get webpage corresponding to molecule"""
        self.page_status=False
        if self.cas==-1:
            return
        for cas in self.cas:
            #Create URL
            self.url="https://webbook.nist.gov/cgi/cbook.cgi?ID=C{0}&Units=SI&Mask=4#Thermo-Phase".format(cas)
            
            #Request page
            self.page=requests.get(self.url)
            
            #Check if "Registry Number Not Found" is in page
            if "Registry Number Not Found" in self.page.text:
                continue
            else:
                self.page_status=True
                break
    
    def _clean_value_string(self,value: str):
        """
        This method is used to remove the ± symbol in the NIST phase data table and return the mean value before the sign
        
        Parameters
        ----------
        value : str
            Raw string from NIST phase data table

        Returns
        -------
        final_value : str
            Mean value before the ± symbol

        """
        if "±" in value:
            final_value=value.split(" ")[0].strip()
        else:
            final_value=value
            
        return final_value
    
    def get_critical_quantities(self, silent_request: bool = True):
        """
        Get critical pressure, temperature, and volume from NIST
        
        Parameters
        ----------
        silent_request : bool, optional
            DESCRIPTION. The default is True. If True, an exception is raised if the correct page is not returned on NIST. If False, a dictionary with a blank "name" and all values equal to -1 is returned.

        Returns
        -------
        dict
            A dictionary with four keys: "name", critical pressure "P_c" in bar, critical temperature "T_c" in K, critical volume "V_c" in l, and critical density "rho_c" in mol/l.

        """
        self._get_phase_page()
        
        # Initialize critical quantity variables
        self.P_c=-1
        self.T_c=-1
        self.V_c=-1
        self.rho_c=-1
        
        # Check if page was successfully retrieved
        if not self.page_status:
            if silent_request:
                return {"name":"","P_c":self.P_c,"V_c":self.V_c,"T_c":self.T_c,"rho_c":self.rho_c}
            else:
                raise Exception("CAS Number not found!")
        
        if "Phase change data" in self.page.text:
        
            # Use BeautifulSoup to get data
            soup=BeautifulSoup(self.page.content,features="lxml")
            
            #Remove final symbol table
            symbol_table=soup.find("table",class_="symbol_table")
            try:
                symbol_table.decompose()
            except AttributeError:
                pass
                
            
            #Get name of compound (for verification)
            name=soup.find("h1",id="Top").text
            
            
            #Find all rows
            rows=soup.find_all("tr")
            
            #In each row, check if Tc, Pc, or Vc are present
            T_check=False
            P_check=False
            V_check=False
            rho_check=False
            for row in rows:
                cells=row.find_all("td")
                try:             
                    first_cell=cells[0].text
                    second_cell=cells[1].text
                except IndexError:
                    continue               
                if "Tc" in first_cell and not T_check:
                    self.T_c=float(self._clean_value_string(second_cell))
                    T_check=True
                elif "Pc" in first_cell and not P_check:
                    self.P_c=float(self._clean_value_string(second_cell))
                    P_check=True
                elif "Vc" in first_cell and not V_check:
                    self.V_c=float(self._clean_value_string(second_cell))
                    V_check=True
                elif "ρc" in first_cell and not rho_check:
                    self.rho_c=float(self._clean_value_string(second_cell))
                    rho_check=True
                
            #Return quantities
            return {"name":name,"P_c":self.P_c,"V_c":self.V_c,"T_c":self.T_c,"rho_c":self.rho_c}
        
        else:
            {"name":"","P_c":self.P_c,"V_c":self.V_c,"T_c":self.T_c,"rho_c":self.rho_c}
        

def get_critical_quantities(molecule_identifier: str):
    """
    Get critical quantities of molecule with the given identifier

    Parameters
    ----------
    molecule_identifier : str
        Molecule identifier can be the common name, IUPAC name, CAS number or SMILES string of the molecule

    Returns
    -------
    critical_dict : dict
        A dictionary with four keys: "name", critical pressure "P_c" in bar, critical temperature "T_c" in K, critical volume "V_c" in l, and critical density "rho_c" in mol/l.
    """
    # Create NISTData object
    mol=NISTData(molecule_identifier)
    
    # Get dictionary of critical quantities 
    critical_dict = mol.get_critical_quantities(silent_request=True)
    
    # Return dictionary
    return critical_dict
        
        
class MolGraph:
    def __init__(self,molecule_smiles: str, node_vec_len : int, max_atoms: int = None):
        """
        Construct a molecular graph of a given molecule with a SMILES string with a node matrix that has dimensions (max_atoms,node_vec_len) and an adjacency matrix with dimensions (max_atoms,max_atoms).
        
        Parameters
        ----------
        molecule_smiles : str
            SMILES string of the molecule
        node_vec_len : int
            DESCRIPTION.
        max_atoms : int, optional
            DESCRIPTION. The default is None.
        """
        
        self.smiles = molecule_smiles
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        self.smiles_to_mol()
        if self.mol is not None:
            self.smiles_to_graph()
            
        
    def smiles_to_mol(self):
        """
        Converts smiles string to Mol object in RDKit
        """
        
        # Use MolFromSmiles from RDKit to get molecule object
        mol=Chem.MolFromSmiles(self.smiles)
        if mol is None:
            self.mol = None
            return
        
        # Add hydrogens to molecule
        self.mol=Chem.AddHs(mol)

        
    def smiles_to_graph(self):
        """
        Converts smiles to a graph with a one-hot encoded node matrix and an adjacency matrix
        """
        
        # Get list of atoms in molecule
        atoms = self.mol.GetAtoms()
        
        # Create empty node matrix
        if self.max_atoms is None:
            n_atoms = len(list(atoms))
        else:
            n_atoms = self.max_atoms
        node_mat = np.zeros((n_atoms, self.node_vec_len))
        
        # Iterate over atoms and add to node matrix
        for atom in atoms:
            # Get atom index and atomic number
            atom_index = atom.GetIdx()
            atom_no = atom.GetAtomicNum()
            
            # Assign to node matrix
            node_mat[atom_index, atom_no] = 1    
        
        # Create empty adjacency matrix
        adj_mat = np.zeros((n_atoms, n_atoms))
        
        # Iterate over bonds and add to adjacency matrix
        bonds = self.mol.GetBonds()
        for bond in bonds:
            # Get indices of atoms in bond
            atom_1 = bond.GetBeginAtomIdx()
            atom_2 = bond.GetEndAtomIdx()
            
            # Add to adjancecy matrix
            adj_mat[atom_1, atom_2] = 1
            adj_mat[atom_2, atom_1] = 1
        
        # Add an identity matrix to adjacency matrix
        # This will make an atom its own neighbor
        adj_mat = adj_mat + np.eye(n_atoms)
        
        # Save both matrices
        self.node_mat = node_mat
        self.adj_mat = adj_mat            
        
def name_to_smiles(name: str):
    """
    Convert common or IUPAC name of a molecule to its SMILES string.

    Parameters
    ----------
    name : str
        Common or IUPAC name of a molecule

    Returns
    -------
    smile : str
        SMILES string of the molecule

    """
    # Convert molecule name to smiles using cirpy
    smile=cirpy.resolve(name,"smiles")
    return smile
                    

def collate_graph_dataset(dataset: Dataset):
    """
    Collate function for the PhaseDataset dataset.

    Parameters
    ----------
    dataset : PhaseDataset
        Object of the PhaseDataset class.

    Returns
    -------
    node_mats_tensor, adj_mats_tensor : tuple of two torch.Tensor objects
        Node matrices with dimensions (batch_size, max_atoms, node_vec_len) and adjacency matrices with dimensions (batch_size, max_atoms, max_atoms)
    outputs_tensor : torch.Tensor with dimensions (batch_size, n_outputs)
        Tensor containing outputs.
    smiles : list
        List of size batch_size containing SMILES strings.
    """
    
    # Create empty lists of node and adjacency matrices, outputs, and smiles
    node_mats = []
    adj_mats = []
    outputs = []
    smiles = []
    
    # Iterate over list and assign each component to the correct list
    for i in range(len(dataset)):
        (node_mat,adj_mat), output, smile = dataset[i]
        node_mats.append(node_mat)
        adj_mats.append(adj_mat)
        outputs.append(output)
        smiles.append(smile)
    
        
    # Create tensors
    node_mats_tensor = torch.cat(node_mats, dim=0)
    adj_mats_tensor = torch.cat(adj_mats, dim=0)
    outputs_tensor = torch.stack(outputs, dim=0)
    
    # Return tensors
    return (node_mats_tensor, adj_mats_tensor), outputs_tensor, smiles


class PhaseDataset(Dataset):
    def __init__(self,critical_property: str, node_vec_len: int, max_atoms: int):
        """
        PhaseDataset class inheriting from the Dataset class in PyTorch. 

        Parameters
        ----------
        critical_property : str
            Critical property ("P_c", "T_c", "V_c", "rho_c")
        node_vec_len : int
            Node vector length of molecular graphs
        max_atoms : int
            Maximum number of atoms in molecular graphs
        """
        
        # Save attributes
        self.critical_property = critical_property
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms

        # Open clean datafile
        curr_dir_path = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv("{0}/clean_datafiles/clean-{1}.csv".format(curr_dir_path,self.critical_property))
        
        # Create lists
        self.indices = df.index.to_list()
        self.names = df["name"].to_list()
        self.smiles = df["smiles"].to_list()
        self.outputs = df[self.critical_property].to_list()
        
    def __len__(self):
        """
        Get length of the dataset

        Returns
        -------
        Length of dataset
        """
        return len(self.indices)
    
    def __getitem__(self,i):
        """
        Returns node matrix, adjacency matrix, output, and SMILES string of molecule at index i

        Parameters
        ----------
        i : int
            Dataset index

        Returns
        -------
        node_mat : torch.Tensor with dimension (max_atoms,node_vec_len)
            Node matrix
        adj_mat: torch.Tensor with dimension (max_atoms,max_atoms)
            Adjacency matrix
        output : torch.Tensor with dimension n_outputs
            Output vector
        smile : str
            SMILES string of molecule
        """
        
        # Get smile
        smile = self.smiles[i]
        
        # Create MolGraph object
        mol = MolGraph(smile, self.node_vec_len, self.max_atoms)
        
        # Get matrices
        node_mat = torch.Tensor(mol.node_mat)
        adj_mat = torch.Tensor(mol.adj_mat)
        
        # Get output
        output = torch.Tensor([self.outputs[i]])
        
        return (node_mat, adj_mat), output, smile
    
class PhaseMolecule(Dataset):
     def __init__(self, smiles: str, critical_property: str, node_vec_len: int, max_atoms: int):
         """
         Class inheriting from the Dataset class in PyTorch. Unlike the PhaseDataset class, this only handles one molecule at a time (instead of batches)

        
        Parameters
        ----------
        smile: str
            SMILES string of molecule
        critical_property : str
            Critical property ("P_c", "T_c", "V_c", "rho_c")
        node_vec_len : int
            Node vector length of molecular graphs
        max_atoms : int
            Maximum number of atoms in molecular graphs

         """
         
         # Save attributes
         self.critical_property = critical_property
         self.node_vec_len = node_vec_len
         self.max_atoms = max_atoms
         self.smiles = smiles
         
     def __len__(self):
         """
         Return length of dataset (in this case, just 1)

         Returns
         -------
         1
         """
         return 1
     
     def __getitem__(self,i):
         """
         Return node matrix, adjacency matrix, and smile of the molecule
         Parameters
         ----------
         i : int
             Dataset index

          Returns
          -------
          node_mat : torch.Tensor with dimension (max_atoms,node_vec_len)
              Node matrix
          adj_mat: torch.Tensor with dimension (max_atoms,max_atoms)
              Adjacency matrix
          smile : str
              SMILES string of molecule
         """
         
         # Get smile
         smile = self.smiles
         
         # Create MolGraph object
         mol = MolGraph(smile, self.node_vec_len, self.max_atoms)
         
         # Get matrices
         node_mat = torch.unsqueeze(torch.Tensor(mol.node_mat),0)
         adj_mat = torch.unsqueeze(torch.Tensor(mol.adj_mat),0)
         
         return (node_mat, adj_mat), smile   
        
def get_dataloaders(dataset: Dataset, train_size: Union[int,float], val_size: Union[int,float], batch_size: int):
    """
    Return PyTorch DataLoaders for the training, validation and test sets.

    Parameters
    ----------
    dataset : Dataset
        Object of the PhaseDataset class
    train_size : Union[int,float]
        Size of training set (either as fraction or number of datapoints)
    val_size : Union[int,float]
        Size of validation set (either as fraction or number of datapoints)
    batch_size : int
        Batch size 

    Returns
    -------
    dict
        Dictionary containing the following: training DataLoader "train_loader", validation DataLoader "val_loader", test DataLoader "test_loader", training set size "train_size", validation set size "val_size", test set size "test_size", training set indices "train_indices", validation set indices "val_indices", test set indices "test_indices", training set outputs "train_outputs" 
    """
    
    # Get length of dataset
    dataset_size = len(dataset)
    
    # Get training dataset size
    if train_size < 1:
        train_size = int(train_size * dataset_size)
    elif train_size >= 1:
        if train_size > dataset_size:
            raise Exception("Training dataset size must be less than  total dataset size")
        else:
            train_size = int(train_size)
    
    # Get validation dataset size
    if val_size < 1:
        val_size = int(val_size * dataset_size)
    elif val_size >= 1:
        if val_size > dataset_size - train_size:
            raise Exception("Validation dataset size must be less than total dataset size minus training dataset size")
        else:
            val_size = int(val_size)
            
    # Get test dataset size
    test_size = dataset_size - train_size - val_size
    
    # Create array of indices and sample training, validation and test indices
    indices = np.arange(0, dataset_size, 1)
    train_indices = np.random.choice(indices, size=train_size, replace=False)
    indices_minus_train = np.array(list(set(indices) - set(train_indices)))    
    val_indices = np.random.choice(indices_minus_train , size=val_size, replace=False)
    test_indices = np.array(list(set(indices_minus_train) - set(val_indices)))
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    total_sampler = SubsetRandomSampler(indices)
    
    # Create dataloaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_graph_dataset)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=collate_graph_dataset)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=collate_graph_dataset)
    total_loader = DataLoader(dataset, batch_size=batch_size, sampler=total_sampler, collate_fn=collate_graph_dataset)
    
    # Get training outputs for standardizing 
    train_outputs = []
    for i in train_indices:
        _,o,_ = dataset[i]
        train_outputs.append(o)
    train_outputs = torch.Tensor(train_outputs)
    
    # Return dictionary with dataloders and other information
    return {"train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader, "total_loader": total_loader, "train_size": train_size, "val_size": val_size, "test_size": test_size, "train_indices": train_indices, "val_indices": val_indices, "test_indices": test_indices, "train_outputs": train_outputs}  

    
def save_dataset_as_pickle(list_of_smiles: list, list_of_outputs: list, dataset_name: str, node_vec_len: int=20):
    
    # Create node and adjacency matrices for each molecule
    list_of_node_mats = []
    list_of_adj_mats = []
    for i,smile in enumerate(list_of_smiles):
        mol = MolGraph(smile, node_vec_len=node_vec_len)
        if mol.mol is not None:
            list_of_node_mats.append(mol.node_mat)
            list_of_adj_mats.append(mol.adj_mat)
        
    # Make matrices
    outputs=np.array(list_of_outputs)
    arr_of_node_mats=np.array(list_of_node_mats)
    arr_of_adj_mats=np.array(list_of_adj_mats)
    
    with open("./{0}.pkl".format(dataset_name),"wb") as f:
        pickle.dump([arr_of_node_mats,arr_of_adj_mats,outputs],f)

def make_clean_datafile(raw_csv_file: str, critical_property: str):
    # Import raw datafile
    df_raw = pd.read_csv(raw_csv_file)
    
    # Filter by property
    df = df_raw.loc[df_raw[critical_property]>-1,["name",critical_property]]
    df = df.reset_index(drop=True)
    
    # Add molecule smiles to a column
    df["smiles"] = ""
    for i in range(df.shape[0]):
        name = df.loc[i, "name"]
        smile = name_to_smiles(name)
        if smile is not None and Chem.MolFromSmiles(smile) is not None:
            df.loc[i, "smiles"] = smile  
            
    # Remove molecules with no smiles
    df = df.loc[df["smiles"]!="",:]
    df = df.reset_index(drop=True)
    
    
    # Save csv
    df.to_csv("./clean_datafiles/clean-{0}.csv".format(critical_property),mode="w")

def add_from_VHD_dataset(critical_property: str):
    # Open clean datafile
    df_clean = pd.read_csv("./clean_datafiles/clean-{0}.csv".format(critical_property))
    
    # Collect list of smiles
    list_of_smiles = df_clean["smiles"].to_list()
    
    # Open VHD datafile
    df_vhd = pd.read_excel("./VHD_dataset.xlsx",skiprows=1)
    
    # Go through each compound, get name, get smiles, and check
    # if already present in clean dataset. If not, add it
    for i in range(df_vhd.shape[0]):
        name = df_vhd.loc[i, "name"]
        smile = name_to_smiles(name)
        if smile not in list_of_smiles:
            add_name = name
            add_smile = smile
            add_prop = df_vhd.loc[i, critical_property]
            if critical_property in "P_c":
                add_prop = add_prop * 10
            add_dict = {"name": add_name, "smiles": add_smile, critical_property: add_prop}
            df_clean = df_clean.append(add_dict, ignore_index=True)
            
    # Reindex
    df_clean = df_clean.reset_index(drop=True)
    
    # Save csv
    df_clean.to_csv("./clean_datafiles/clean-{0}.csv".format(critical_property),mode="w")
    

def scrape_NIST_from_ASM(dataset_file,output_file,start_index=0,end_index=-1,timeout=5):
    # Open ASM file
    df=pd.read_excel(dataset_file,sheet_name="Database",skiprows=3)
    
    # Get SMILES strings
    list_of_smiles=df["SMILES"]

    # Create output dataframe
    df_out=pd.DataFrame({"name":[],"P_c":[],"T_c":[],"V_c":[],"rho_c":[]})
                        
    # For smile in list, scrape NIST data
    for smile in tqdm(list_of_smiles[start_index:end_index+1]):
        mol=NISTData(smile)
        c=mol.get_critical_quantities(silent_request=True)
        df_out=df_out.append(c,ignore_index=True)
        time.sleep(timeout)
    
    if start_index > 0:
        write_mode="a"
        header_flag=False
    else:
        write_mode="w"
        header_flag=True
    df_out.to_csv(output_file,mode=write_mode,header=header_flag,index=False)
    
    critical_quantities = ["P_c", "T_c", "V_c", "rho_c"]
    for c in critical_quantities:
        make_clean_datafile(output_file, c)
   
def search_in_dataset(molecule_identifier: str, critical_property: str ):
    # Get smiles
    smiles = cirpy.resolve(molecule_identifier, "smiles")
    if smiles is None:
        raise Exception("Molecule not found!")
    
    # Open dataset
    curr_dir_path=os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(curr_dir_path,"clean_datafiles","clean-{0}.csv".format(critical_property))
    df = pd.read_csv(file_path)
    
    # List of smiles
    list_of_smiles = df["smiles"].to_list()

    # Search in list
    if smiles in list_of_smiles:
        value = df.loc[df["smiles"]==smiles,critical_property].values
        return True,value
    else:
        c = get_critical_quantities(molecule_identifier)
        if c is not None:
            value = c[critical_property]
        else:
            value = -1.
        if value != -1:
            return False, value
        else:
            return False, 0
        
   
#if __name__=="__main__":
    #mol=NISTData("Water")
    #c=mol.get_critical_quantities(silent_request=True)
    #print(c)
    #scrape_NIST_from_ASM("ASM_dataset.xlsx","phase_database.csv",start_index=16140,end_index=16417,timeout=2)
    #make_clean_datafile("phase_database.csv","P_c")
    #data=PhaseDataset("P_c", node_vec_len=40)
    #for i in range(len(data)):
    #    print(data[i])
    #add_from_VHD_dataset("P_c")
    
    
          
      