import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import utils_model as utils
import graph_construct
import kernels
import sparse_with_grad

###########################
#     CORE MODEL CODE     #
###########################

class GNN(nn.Module):
  '''
  Graph neural network:
    - Runs through several graph convolution layers of specified type
    - Performs final classification using logistic regression
  '''
  def __init__(self, nb_hidden, nb_layer, input_dim, spatial_dims=None):
    super(GNN, self).__init__()
    # Initialize GNN layers
    self.create_sparse_graph = graph_construct.MLP_PG(
                                              input_dim=input_dim,
                                              kernel=kernels.MLP(input_dim,1024),
                                              spatial_dims=spatial_dims
                                              )
    # self.create_sparse_graph = Rand_Tree(Gaussian(spatial_dims))
    first_layer = GNN_Layer(
                            input_dim, 
                            nb_hidden, 
                            apply_norm=False
                           )
    rem_layers = [GNN_Layer(nb_hidden, nb_hidden) for _ in range(nb_layer-1)]
    self.layers = nn.ModuleList([first_layer]+rem_layers)
    # Initialize final readout layer
    self.readout_fc = nn.Linear(nb_hidden, 1)
    self.readout_norm = nn.InstanceNorm1d(1)
    self.readout_act = nn.Sigmoid()

  def forward(self, emb, mask, batch_nb_nodes):
    # Create sparse graph where adj is approx nlogn sized and sparse
    '''
    print(emb.size())
    t0 = time.time()
    emb, adj, sum_probs, nb_edges = self.create_sparse_graph(emb.cpu())
    emb = emb.cuda()
    adj = adj.cuda()
    '''
    emb = emb.squeeze(0)
    # Run PG to get adj
    emb, adj, sum_probs, nb_edges = self.create_sparse_graph(emb)
    emb = emb.unsqueeze(0)
    '''
    t1 = time.time()
    '''
    # print(emb.size(), nb_edges)
    batch_size, nb_pts, input_dim  = emb.size()
    # Run through layers
    for i, layer in enumerate(self.layers):
      emb, adj = layer(emb, adj, mask, batch_nb_nodes)
    # Apply final readout and return
    emb = emb.sum(1)
    emb = self.readout_norm(emb.unsqueeze(1)).squeeze(1)
    # Logistic regression
    emb = self.readout_fc(emb)
    emb = self.readout_act(emb).squeeze(1)
    '''
    t2 = time.time()
    print("Time to make adj:  {:.4f}".format(t1-t0))
    print("Time for all else: {:.4f}".format(t2-t1))
    '''
    return emb, sum_probs

class GNN_Layer(nn.Module):
  def __init__(self, input_dim, nb_hidden, kernel=None, apply_norm=True, residual=True):
    super(GNN_Layer, self).__init__()
    self.kernel=kernel
    self.apply_norm=apply_norm
    # Create two graph convolution modules in case residual
    self.convA = Graph_Convolution(input_dim, nb_hidden // 2)
    self.convB = Graph_Convolution(input_dim, nb_hidden // 2)
    self.act = nn.ReLU()
    self.residual = residual

  def forward(self, emb, adj, mask, batch_nb_nodes):
    # Optionally normalize embedding
    if self.apply_norm:
      emb = utils.batch_norm(emb, mask, batch_nb_nodes)
    # Apply convolution
    embA = self.convA(emb, adj)
    embB = self.convB(emb, adj)
    # Apply activations
    embA = self.act(embA)
    # Apply activation to all features if not residual
    if not self.residual:
      embB = self.act(embB)
    # Concatenate features and return
    emb_out = torch.cat((embA, embB), dim=2)
    return emb_out, adj
      
      
class Graph_Convolution(nn.Module):
  def __init__(self, input_dim, nb_hidden):
    super(Graph_Convolution, self).__init__()
    self.fc = nn.Linear(input_dim*2, nb_hidden)

  def forward(self, emb, adj):
    spread = torch.mm(adj, emb.squeeze(0)) # Apply adjacency matrix
    spread = spread.unsqueeze(0)
    spread = torch.cat((spread, emb), 2) # Concatenate with original signal
    emb_out = self.fc(spread) # Apply affine transformation
    return emb_out
    
