import torch
import torch.nn as nn
from torch.autograd import Variable

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
    first_layer = GNN_Layer(
                            input_dim, 
                            nb_hidden, 
                            kernel=GaussianSoftmax(spatial_dims),
                            apply_norm=False
                           )
    rem_layers = [GNN_Layer(nb_hidden, nb_hidden) for _ in range(nb_layer-1)]
    self.layers = nn.ModuleList([first_layer]+rem_layers)
    # Initialize final readout layer
    self.readout_fc = nn.Linear(nb_hidden, 1)
    self.readout_norm = nn.InstanceNorm1d(1)
    self.readout_act = nn.Sigmoid()

  def forward(self, emb, mask, batch_nb_nodes):
    batch_size, nb_pts, input_dim  = emb.size()
    # Create dummy first adjacency matrix
    adj = Variable(torch.ones(batch_size, nb_pts, nb_pts))
    if emb.is_cuda:
      adj = adj.cuda()
    # Run through layers
    for i, layer in enumerate(self.layers):
      emb, adj = layer(emb, adj, mask, batch_nb_nodes)
    # Apply final readout and return
    emb = mask_embedding(emb, mask).sum(1)
    emb = self.readout_norm(emb.unsqueeze(1)).squeeze(1)
    # Logistic regression
    emb = self.readout_fc(emb)
    emb = self.readout_act(emb).squeeze(1)
    return emb

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
    # Optionally update kernel matrix
    if self.kernel is not None:
      adj = self.kernel(emb, mask)
    # Optionally normalize embedding
    if self.apply_norm:
      emb = batch_norm_with_padding(emb, mask, batch_nb_nodes)
    # Ensure no signal on padded nodes
    emb = mask_embedding(emb, mask)
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
    spread = torch.bmm(adj, emb) # Apply adjacency matrix
    spread = torch.cat((spread, emb), 2) # Concatenate with original signal
    emb_out = self.fc(spread) # Apply affine transformation
    return emb_out
    

###################
#     KERNELS     #
###################
class Gaussian(nn.Module):
  '''
  Computes an adjacency matrix by using a Gaussian kernel
  over each pair of nodes within each sample.
  Can be initialized to only use a subset of features
  in the kernel (e.g. spatial coordinates).
  '''
  def __init__(self, spatial_coords=None):
    super(Gaussian, self).__init__()
    self.sigma = nn.Parameter(torch.rand(1) * 0.02 + 0.99)
    self.spatial_coords = spatial_coords

  def _apply_norm_and_mask(self, adj, mask):
    return adj * mask

  def forward(self, emb_in, mask):
    batch, nb_pts, nb_features = emb_in.size()
    # Select only specified features, if given
    if self.spatial_coords is not None:
      emb = emb_in[:,:,self.spatial_coords]
    else:
      emb = emb_in
    # Expand and transpose coordinates
    emb = emb.unsqueeze(2).expand(batch, nb_pts, nb_pts, emb.size()[2])
    emb_t = emb.transpose(1,2)
    # Apply gaussian kernel to adj
    adj = ((emb-emb_t)**2).mean(3)
    adj = torch.exp(-adj.div(self.sigma**2))
    # Normalize over norm (to be used in child classes for e.g. Softmax)
    adj = self._apply_norm_and_mask(adj, mask)
    return adj

class GaussianSoftmax(Gaussian):
  '''
  Exactly identical to the Gaussian kernel, but
  adj matrix is renormalized using Softmax.
  '''
  def __init__(self, spatial_coords=None):
    super(GaussianSoftmax, self).__init__(spatial_coords)

  def _apply_norm_and_mask(self, adj, mask):
    # Softmax applies mask so no need to repeat
    return softmax_with_padding(adj, mask)


###########################
#     MODEL UTILITIES     #
###########################
def mask_embedding(emb, mask):
  '''
  Sets added padding nodes in each sample to zero.
  '''
  nb_features = emb.size()[2]
  return torch.mul(emb, mask[:,:,0:1].repeat(1,1,nb_features))

def mean_with_padding(emb, mask, batch_nb_nodes):
  '''
  Computes mean of embedding for each sample over point cloud for each feature.
  Accounts for zero-padded nodes.
  '''
  summed = mask_embedding(emb, mask).sum(1)
  batch_div_by = batch_nb_nodes.unsqueeze(1).repeat(1,emb.size()[2])
  return summed / (batch_div_by + 10**-20)

def batch_norm_with_padding(emb, mask, batch_nb_nodes):
  '''
  Normalizes each feature within each sample to mean zero, var 1
  '''
  mean = mean_with_padding(emb, mask, batch_nb_nodes)
  emb_centered = emb-mean.unsqueeze(1)

  var = mean_with_padding(emb_centered**2, mask, batch_nb_nodes) + 10**-20
  emb_norm = emb_centered / var.sqrt().unsqueeze(1)
  return emb_norm

def softmax_with_padding(adj, mask):
  '''
  Computes softmax over rows of adj.
  Accounts for point cloud size of samples by
  renormalizing using the mask.
  '''
  S = nn.functional.softmax(adj, dim=2)
  S = S * mask
  E = S.sum(2,keepdim=True) + 10**-20
  return S / E
