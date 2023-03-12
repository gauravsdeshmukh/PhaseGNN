# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 21:58:23 2023

@author: Gaurav Deshmukh
"""

import torch
import torch.nn as nn


#### CLASSES        
class ConvolutionLayer(nn.Module):
    """
    Create a simple graph convolution layer
    """
    
    def __init__(self, node_fea_len: int):
        super(ConvolutionLayer,self).__init__()
        self.conv_linear = nn.Linear(node_fea_len, node_fea_len)
        self.conv_activation = nn.ReLU()
        
    def forward(self, node_mat, adj_mat):
        """
        Parameters
        ----------
        node_mat : (n_atoms, node_vec_len) array
            Node matrix
        adj_mat : (n_atoms, n_atoms) array
            Adjacency matrix

        Returns
        -------
        node_fea : (n_atoms, node_vec_len) 
            Updated node features
        """
        # Calculate number of neighbors
        n_neighbors = adj_mat.sum(dim = -1, keepdims = True)
        
        # Perform linear transformation to node features
        node_fea = self.conv_linear(node_mat)
        
        # Perform matrix multimplication with adjacency matrix 
        node_fea = torch.bmm(adj_mat,node_fea)
        
        # Divide by number of neighbors
        node_fea = node_fea / n_neighbors
        
        # Apply activation function
        node_fea = self.conv_activation(node_fea)
        
        return node_fea
    

class PoolingLayer(nn.Module):
    """
    Create a pooling layer to average node-level properties into graph-level properties
    """
    def __init__(self):
        super(PoolingLayer,self).__init__()
        
    def forward(self, node_fea):
        pooled_node_fea = node_fea.mean(dim=1)
        return pooled_node_fea        

class PhaseGNN(nn.Module):
    """
    Create a graph neural network to predict phase change properties of molecules
    """
    def __init__(self, node_vec_len: int, node_fea_len: int, hidden_fea_len: int, n_conv: int, n_hidden: int, n_outputs: int, p_dropout: float = 0.):
        """
        Class for the PhaseGNN model

        Parameters
        ----------
        node_vec_len : int
            Node vector length
        node_fea_len : int
            Node feature length
        hidden_fea_len : int
            Hidden feature length (number of nodes in hidden layer)
        n_conv : int
            Number of convolution layers
        n_hidden : int
            Number of hidden layers
        n_outputs : int
            Number of outputs
        p_dropout : float, optional
            Probability (0<=p_dropout<1) that a node is dropped out. The default is 0.. 

        """

        super(PhaseGNN,self).__init__()
        
        # Define hyperparameters
        self.n_hidden = n_hidden
        self.n_conv = n_conv
        self.node_vec_len = node_vec_len
        self.node_fea_len = node_fea_len
        self.hidden_fea_len = hidden_fea_len
        self.n_outputs = n_outputs
        self.p_dropout = p_dropout

        # Define layers
        # Initial transformation from node matrix to node features
        self.init_transform = nn.Linear(node_vec_len, self.node_fea_len)
        
        # Convolution layers
        self.conv_layers = nn.ModuleList([ConvolutionLayer(node_fea_len=self.node_fea_len) for i in range(self.n_conv)])
        
        # Pool convolution outputs
        self.pooling = PoolingLayer()
        self.pooled_node_fea_len = self.node_fea_len
        
        # Pooling activation
        self.pooling_activation = nn.ReLU()
        
        # From pooled vector to hidden layers
        self.pooled_to_hidden = nn.Linear(self.pooled_node_fea_len, self.hidden_fea_len)
        
        # Hidden layer
        self.hidden_layer = nn.Linear(self.hidden_fea_len,self.hidden_fea_len)
        
        # Hidden layer activation function
        self.hidden_activation = nn.ReLU()
        
        # Hidden layer dropout
        self.dropout = nn.Dropout(p = p_dropout)
        
        # If hidden layers more than 1, add more hidden layers
        if n_hidden > 1:
            self.hidden_layers = nn.ModuleList([self.hidden_layer for i in range(n_hidden - 1)])
            self.hidden_activation_layers = nn.ModuleList([self.hidden_activation for i in range(n_hidden -1)])
            self.hidden_dropout_layers = nn.ModuleList([self.dropout for i in range(n_hidden-1)])
            
        # Final layer going to the output
        self.hidden_to_output = nn.Linear(self.hidden_fea_len, self.n_outputs)
        
    
    def forward(self,node_mat,adj_mat):       
        """
        Forward pass

        Parameters
        ----------
        node_mat : torch.Tensor with shape (batch_size, max_atoms, node_vec_len)
            Node matrices
        adj_mat : torch.Tensor with shape (batch_size, max_atoms, max_atoms)
            Adjacency matrices

        Returns
        -------
        out : torch.Tensor with shape (batch_size, n_outputs)
            Output tensor
        """
        # Perform initial transform on node_mat
        node_fea = self.init_transform(node_mat)
        
        # Perform convolutions
        for conv in self.conv_layers:
            node_fea = conv(node_fea,adj_mat)
            
        # Perform pooling
        pooled_node_fea = self.pooling(node_fea)
        pooled_node_fea = self.pooling_activation(pooled_node_fea)        
        
        # First hidden layer
        hidden_node_fea = self.pooled_to_hidden(pooled_node_fea)
        hidden_node_fea = self.hidden_activation(hidden_node_fea)
        hidden_node_fea = self.dropout(hidden_node_fea)
        
        # Subsequent hidden layers
        if self.n_hidden > 1:
            for i in range(self.n_hidden-1):
                hidden_node_fea = self.hidden_layers[i](hidden_node_fea)
                hidden_node_fea = self.hidden_activation_layers[i](hidden_node_fea)
                hidden_node_fea = self.hidden_dropout_layers[i](hidden_node_fea)
                
        # Output
        out = self.hidden_to_output(hidden_node_fea)
        
        return out
    
    def return_pooled_features(self,node_mat,adj_mat):
        """
        Return pooled feature vector generated after convolution and pooling layers

        Parameters
        ----------
        node_mat : torch.Tensor with shape (batch_size, max_atoms, node_vec_len)
            Node matrices
        adj_mat : torch.Tensor with shape (batch_size, max_atoms, max_atoms)
            Adjacency matrices

        Returns
        -------
        pooled_node_fea : torch.Tensor with shape (batch_size, node_fea_len)
            Pooled feature vector (embedding)

        """
        # Perform initial transform on node_mat
        node_fea = self.init_transform(node_mat)
        
        # Perform convolutions
        for conv in self.conv_layers:
            node_fea = conv(node_fea,adj_mat)
            
        # Perform pooling
        pooled_node_fea = self.pooling(node_fea)
        pooled_node_fea = self.pooling_activation(pooled_node_fea)
        
        return pooled_node_fea
      

#if __name__=="__main__":
    # df_raw=pd.read_csv("phase_database.csv")
    # df=df_raw.loc[df_raw["T_c"]>-1,:]
    # df=df.reset_index()
    
    # # Try for one molecule
    # name=df.loc[100,"name"]
    # smile=cirpy.resolve(name,"smiles")
    # g=MolGraph(smile)
    # n=torch.Tensor(g.node_mat).view(1,g.node_mat.shape[0],g.node_mat.shape[1])
    # a=torch.Tensor(g.adj_mat).view(1,g.adj_mat.shape[0],g.adj_mat.shape[1])
    # model=PhaseGNN(node_vec_len=20, node_fea_len=20, hidden_fea_len=10, n_conv=2, n_hidden=2, n_outputs=1)
    

    # with torch.no_grad():
    #     out = model(n,a)
    #     print(out)
    
    # list_of_smiles=[]
    # list_of_outputs=[]
    # for i in tqdm(range(df.shape[0])):
    #     _name=df.loc[i,"name"]
    #     _smile=name_to_smiles(_name)
    #     _Tc=df.loc[i,"T_c"]
    #     if isinstance(_smile,str):
    #         list_of_smiles.append(_smile)
    #         list_of_outputs.append(float(_Tc))
    #         time.sleep(1)          
    # create_dataset(list_of_smiles,list_of_outputs, "critical_temp_20230307")