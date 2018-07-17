import torch
import torch.nn as nn
from torch.autograd import Variable

##########################
#     NLOGN TREE GNN     #
##########################
class Division_Tree(nn.Module):
  '''
  Recursively divide the point cloud input into subsets.
  Build binary tree associated with splits.
  Permute point cloud s.t. binary tree divides point cloud around a pivot
  at each layer of tree.
  Leaf nodes of tree store adj matrix for subset of nodes.
  '''
  def __init__(self, kernel, min_nodes=30, max_depth=10):
    super(Division_Tree, self).__init__()
    self.min_nodes = min_nodes
    self.max_depth = max_depth
    self.kernel = kernel

  def forward(self, X):
    all_nodes, i, v = self.dfs(X, depth=1)
    batch, nb_nodes, nb_features = all_nodes.size()
    if X.is_cuda:
      print("Sparse not implemented yet for cuda")
      exit()
    else:
      tensor = torch.sparse.FloatTensor
    adj = tensor(i, v, torch.Size([nb_nodes, nb_nodes]))
    return all_nodes, adj

  def _leaf_reached(self, nb_nodes, depth):
    if (nb_nodes <= self.min_nodes) or (depth == self.max_depth):
      return True
    else:
      return False

  def _divide(self, X):
    '''
    Takes as input the point cloud X.
    Returns two point clouds, left and right.
    '''
    raise TypeError("Division Tree is abstract class. Child must implement _divide")

  def _cartesian(self, X):
    '''
    Compute cartesian product of point cloud X, originally size n x m.
    Returns cartesian product of size n^2 x 2m.
    '''
    batch, nb_node, fmap = X.size()
    a = X.repeat(1,1,nb_node).resize(batch, nb_node*nb_node, fmap)
    b = X.repeat(1,nb_node,1)
    c = torch.cat((a,b),2)
    return c

  def _get_dense_adj_indices(self, nb_nodes):
    i = torch.arange(0,nb_nodes).type(torch.LongTensor)
    i = i.resize_(nb_nodes,1)
    a = i.repeat( 1,nb_nodes).resize_(nb_nodes*nb_nodes, 1).squeeze(1)
    b = i.repeat( nb_nodes, 1).squeeze(1)
    i = torch.cat((a.unsqueeze(0),b.unsqueeze(0)),0)
    return i

  def dfs(self, X, depth):
    batch, nb_nodes, nb_features = X.size()
    if self._leaf_reached(X.size(1), depth):
      # Compute dense adjacency matrix
      emb = self._cartesian(X)
      emb_l, emb_r = emb.chunk(2, 2)
      # Get dense edges for all leaf vertices
      v = self.kernel(emb_l, emb_r).data.squeeze(0)
      # Get dense edge indices
      i = self._get_dense_adj_indices(nb_nodes)
      return X, i, v
    else:
      l_nodes, r_nodes = self._divide(X)
      l_nodes, l_i, l_v = self.dfs(l_nodes, depth+1)
      r_nodes, r_i, r_v = self.dfs(r_nodes, depth+1)
      # Update right indices to reflect all nodes size
      r_i += l_nodes.size(1)
      all_nodes = torch.cat((l_nodes, r_nodes), dim=1)
      # Combine adj indices and values from left, right
      i = torch.cat((l_i, r_i), 1)
      v = torch.cat((l_v, r_v))
      return all_nodes, i, v

class Rand_Tree(Division_Tree):
  def __init__(self, kernel):
    super(Rand_Tree, self).__init__(kernel)

  def _divide(self, X):
    nb_nodes = X.size(1)
    node_idx = torch.randperm(nb_nodes)
    if X.is_cuda:
      node_idx = node_idx.cuda()
    pivot = nb_nodes // 2
    l_idx = node_idx[:pivot]
    r_idx = node_idx[pivot:]
    l_nodes = X[:,l_idx]
    r_nodes = X[:,r_idx]
    return l_nodes, r_nodes
  
class KMeans_Tree(Division_Tree):
  def __init__(self, kernel):
    super(KMeans_Tree, self).__init__(kernel)
    self.kmeans = KMeans(n_clusters=2, n_init=4, max_iter=30)

  def _divide(self, X):
    nb_nodes = X.size(1)
    # Prepare for Scikit handling
    # Only use spatial coordinates
    X_kmeans = X[:,:,SPATIAL_COORDS].data.squeeze(0).cpu().numpy()
    # Predict classes
    kmeans_classes = self.kmeans.fit_predict(X_kmeans)
    # Bring classes to torch and product split
    class_0 = np.argwhere(kmeans_classes==0).squeeze(1)
    class_1 = np.argwhere(kmeans_classes==1).squeeze(1)
    l_idx = torch.LongTensor(class_0)
    r_idx = torch.LongTensor(class_1)
    l_nodes = X[:,l_idx]
    r_nodes = X[:,r_idx]
    return l_nodes, r_nodes

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
    self.create_sparse_graph = Rand_Tree(Gaussian(spatial_dims))
    first_layer = GNN_Layer(
                            input_dim, 
                            nb_hidden, 
                            kernel=Gaussian(spatial_dims),
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
    emb, adj = self.create_sparse_graph(emb)
    batch_size, nb_pts, input_dim  = emb.size()
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
    spread = torch.mm(adj, emb.squeeze(0)) # Apply adjacency matrix
    spread = spread.unsqueeze(0)
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
  Input vectors assumed to be size n x m, where m is dimension of one point.
  '''
  def __init__(self, spatial_coords=None):
    super(Gaussian, self).__init__()
    self.sigma = 100.0
    self.spatial_coords = spatial_coords

  def forward(self, emb_in_0, emb_in_1):
    batch, nb_pts, nb_features = emb_in_0.size()
    # Select only specified features, if given
    if self.spatial_coords is not None:
      emb_0 = emb_in_0[:,:,self.spatial_coords]
      emb_1 = emb_in_1[:,:,self.spatial_coords]
    else:
      emb_0 = emb_in_0
      emb_1 = emb_in_1
    # Apply gaussian kernel to adj
    adj = ((emb_0-emb_1)**2).mean(2)
    adj = torch.exp(-adj.div(self.sigma**2))
    return adj


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
