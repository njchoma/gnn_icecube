import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

if torch.cuda.is_available():
  t_type = torch.cuda
else:
  t_type = torch

####################
#     NLOGN PG     #
####################
class PG_Tree(nn.Module):
  '''
  Module which recursively divides the input points into two clusters.
  Nodes are added at each division point which connect to all nodes in 
    the respective clusters and to all parent added nodes.
  The output is an augmented point cloud, sparse adjacency matrix, and 
    the sum of probabilities of all paths taken.
  '''
  def __init__(self, input_dim, max_nodes=10, max_depth=10):
    '''
    max_nodes is the maximum number of nodes which can belong to a cluster.
    max_depth is the max level of recursion. Prioritized over max_nodes.
    '''
    super(PG_Tree, self).__init__()
    self.max_nodes = max_nodes
    self.max_depth = max_depth
    self.nb_new_features = input_dim-1 # no indicator yet for real/fake nodes
    self.pg_mlp = PG_MLP((input_dim-1)+self.nb_new_features, nb_hidden=32)
    self.batch_norm = nn.BatchNorm1d(input_dim-1)

  def forward(self, X_in):
    '''
    X_in is assumed to be shape [nb_points, nb_features].
    '''
    # Initialize for DFS
    self._initialize_dfs(X_in)
    X_norm = self.batch_norm(X_in) # IMPORTANT FOR PG!!
    # Build X features and initial subtree (contains all nodes)
    pg_mask = make_ones(self.nb_nodes, cuda=X_in.is_cuda)
    full_tree = Subtree(t_type.LongTensor(np.arange(self.nb_nodes)),
                        self.max_nodes, t_type.LongTensor([]))
    # DFS
    pg_sum = self.dfs(X_norm, [full_tree], pg_mask, depth=0)
    # Prepare augmented nodes
    X_updated = self.gather_nodes(X_norm)
    # Set adjacency edge values
    v = self.set_edge_values(X_updated)
    # Make sparse adj
    nb_nodes = X_updated.size(0)
    adj = t_type.sparse.FloatTensor(self.indices, v, torch.Size([nb_nodes, nb_nodes]))
    return X_updated, adj, pg_sum, v.size(0)

  def _initialize_dfs(self, X_in):
    '''
    Make adjacency matrix lists and set next augmented point index.
    '''
    self.nb_nodes, self.nb_features = X_in.size()
    self.next_idx = self.nb_nodes
    self.new_nodes = []
    self.indices = t_type.LongTensor([])

  def dfs(self, X_in, subtrees, pg_mask, depth):
    '''
    '''
    for s in subtrees:
      # Set aggregate (e.g. mean, var) information
      s.set_features(X_in)
      # Build new points for each subtree
      self.create_new_node(X_in, s)
    # Add aggregate features to input nodes
    agg_feats = make_zeros((self.nb_nodes, self.nb_new_features), cuda=X_in.is_cuda)
    for s in subtrees:
      agg_feats[s.node_indices] = s.features
    all_feats = torch.cat((X_in, agg_feats),dim=1)
    # Make new predictions
    splits, pg_sum = self.make_predictions(all_feats, pg_mask)
    # Split subtrees and track subtrees which become leaves (dense)
    new_subtrees, dense_subtrees = self.split_subtrees(splits, subtrees)
    # Final connections to and within dense subtrees
    for d in dense_subtrees:
      self.create_new_node(X_in, d) # make one final node for cluster
      self.make_dense(X_in, d) # Add dense in-cluster connections
      pg_mask = self.update_mask(d, pg_mask)
    # DFS again if criteria met
    if (depth < self.max_depth) and (len(new_subtrees) != 0):
      pg_sum += self.dfs(X_in, new_subtrees, pg_mask, depth+1)
    return pg_sum

  def split_subtrees(self, splits, subtrees):
    '''
    Split subtrees into further nodes. Track which subtrees become leaves.
    '''
    new_subtrees = []
    dense_subtrees = []
    for s in subtrees:
      s_new, d_new = s.split_subtree(splits)
      new_subtrees.extend(s_new)
      dense_subtrees.extend(d_new)
    return new_subtrees, dense_subtrees

  def set_edge_values(self, X):
    '''
    Get edge values between index pairs
    '''
    # Get index pairs (for later use)
    idx = self.indices
    nb_edges = idx.size(1)
    idxA = idx[0].reshape(nb_edges, 1)
    idxB = idx[1].reshape(nb_edges, 1)
    idx_pairs = torch.cat((idxA, idxB),dim=1)
    # Use binary edges for now since can't learn in kernel for sparse adj.
    v = make_ones(nb_edges, cuda=X.is_cuda)
    return v

  def gather_nodes(self, X_in):
    '''
    Put indicater variable on augmented nodes and add to X_in
    '''
    # Put zeros on real nodes
    zeros = make_zeros((X_in.size(0),1), cuda=X_in.is_cuda)
    X = torch.cat((X_in, zeros),dim=1)
    # Put ones on augmented nodes
    X_aug = torch.cat(self.new_nodes,dim=0)
    ones = make_ones((X_aug.size(0),1), X_in.is_cuda)
    X_aug = torch.cat((X_aug, ones),dim=1)
    # Concatenate all nodes
    X_all = torch.cat((X, X_aug),dim=0)
    return X_all

  def make_dense(self, X_in, leaf):
    '''
    Make dense connections for all nodes within leaf cluster
    '''

    n = leaf.node_indices.size(0)
    a = leaf.node_indices.unsqueeze(1)
    a = a.repeat(1,n).resize_(1,n*n)
    b = leaf.node_indices.repeat(n).resize_(1,n*n)
    c = torch.cat((a,b),0)
    self.add_new_edges(c)


  def update_mask(self, leaf, pg_mask):
    '''
    Mask nodes which are at leaf level
    '''
    pg_mask[leaf.node_indices] = 0
    return pg_mask

  def make_predictions(self, all_feats, pg_mask):
    '''
    Compute left / right probabilities on X_in.
    Random sample to produce splits.
    Sum probabilities for each class.
    '''
    probs = self.pg_mlp(all_feats)[:,0]
    splits = torch.bernoulli(probs)
    # Sum probabilities for all splits for policy gradient
    one_class = (splits > 0.5).nonzero()[:,0]
    zero_class = (splits < 0.5).nonzero()[:,0]
    # Mask probabilities for any nodes which are already in a leaf
    one_class_mask = pg_mask[one_class]
    zero_class_mask = pg_mask[zero_class]
    one_probs = probs[one_class] * one_class_mask
    zero_probs = (1-probs[zero_class]) * zero_class_mask
    pg_sum = one_probs.sum() + zero_probs.sum()
    return splits, pg_sum

  def create_new_node(self, nodes, subtree):
    '''
    Add new node to graph and make new edges.
    '''
    # Create new node (just use mean for now)
    new_node = nodes[subtree.node_indices]
    new_node = new_node.mean(dim=0, keepdim=True)
    node_idx = t_type.LongTensor([self.next_idx])
    # Add new node to graph and update next available index
    self.new_nodes.append(new_node)
    self.next_idx += 1
    # Add new node index to subtree's parent list
    subtree.parent_nodes = torch.cat((subtree.parent_nodes, node_idx))
    # Add connections from new node to subtree nodes and parents
    i0 = torch.cat((subtree.node_indices, subtree.parent_nodes)).unsqueeze(0)
    i1 = node_idx.expand_as(i0)
    # Add edges in both directions
    iA = torch.cat((i0, i1),dim=0)
    iB = torch.cat((i1, i0),dim=0)
    i = torch.cat((iA, iB),dim=1)
    self.add_new_edges(i)

  def add_new_edges(self, i):
    '''
    Add new edges to the graph
    '''
    self.indices = torch.cat((self.indices, i),dim=1)

class Subtree(object):
  '''
  Holds information about the current subtree including:
    - bool if leaf reached
    - list of current nodes
    - list of parent nodes
  '''
  def __init__(self, node_indices, max_nodes, parent_nodes):
    self.node_indices = node_indices
    self.parent_nodes = parent_nodes
    self.max_nodes = max_nodes
    self.leaf = True if len(node_indices) < max_nodes else False

  def set_features(self, X_in):
    '''
    Set aggregated features for this subtree (e.g. mean, variance).
    '''
    nodes = X_in[self.node_indices]
    mean = nodes.mean(dim=0)
    self.features = mean

  def split_subtree(self, splits):
    '''
    Split current subtree into two further subtrees
    '''
    subtree_splits = splits[self.node_indices]
    l_split = (subtree_splits < 0.5).nonzero()
    r_split = (subtree_splits > 0.5).nonzero()
    subN = [] # new
    subD = [] # dense
    try:
      l_nodes = self.node_indices[l_split].squeeze(1)
      l_subtree = Subtree(l_nodes, self.max_nodes, self.parent_nodes)
      subD.append(l_subtree) if l_subtree.leaf else subN.append(l_subtree)
    except:
      pass
    try:
      r_nodes = self.node_indices[r_split].squeeze(1)
      r_subtree = Subtree(r_nodes, self.max_nodes, self.parent_nodes.clone())
      subD.append(r_subtree) if r_subtree.leaf else subN.append(r_subtree)
    except:
      pass
    return subN, subD

class PG_MLP(nn.Module):
  '''
  One-layer MLP.
  '''
  def __init__(self, nb_input, nb_hidden):
    super(PG_MLP, self).__init__()
    self.fc1 = nn.Linear(nb_input, nb_hidden)
    self.act1 = nn.SELU()
    self.fc2 = nn.Linear(nb_hidden, 1)
    self.act2 = nn.Sigmoid()

  def forward(self, X_in):
    h1 = self.act1(self.fc1(X_in))
    out = self.act2(self.fc2(h1))
    return out




##########################
#     NLOGN TREE GNN     #
##########################
class Division_Tree(nn.Module):
  '''
  Recursively divide the point cloud input into subsets, traversing implicit
  binary tree.
  Add dense edges at leaf level.
  At inner tree nodes, add new point which then connects to all
  points within subtree.
  Then construct sparse adjacency matrix for O(nlogn) complexity.
  '''
  def __init__(self, kernel, min_nodes=40, max_depth=10):
    super(Division_Tree, self).__init__()
    self.min_nodes = min_nodes
    self.max_depth = max_depth
    self.kernel = kernel
    self.pad = nn.ConstantPad1d((0,1),0.0)

  def forward(self, X):
    # Append zero-padding to indicate true node
    all_nodes = self.pad(X)
    all_nodes, i, v = self.dfs(all_nodes, depth=1)
    batch, nb_nodes, nb_features = all_nodes.size()
    if X.is_cuda:
      tensor = torch.cuda.sparse.FloatTensor
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
    a = X.repeat(1,1,nb_node).resize_(batch, nb_node*nb_node, fmap)
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

  def _get_sparse_adj_indices(self, nb_nodes):
    i = torch.arange(0,nb_nodes).type(torch.LongTensor)
    new_node = torch.LongTensor([nb_nodes]).repeat(nb_nodes)
    i = torch.cat((i.unsqueeze(0),new_node.unsqueeze(0)),0)
    i_prime = torch.cat((i[1:2], i[0:1]),0)
    i = torch.cat((i, i_prime),1)
    i = torch.cat((i, torch.LongTensor([[nb_nodes],[nb_nodes]])),1)
    return i

  def _get_new_node(self, X):
    '''
    Takes point cloud as input and returns a new node which
    somehow summarizes the point cloud (in this case the mean).
    '''
    new_node = X.mean(1,keepdim=True)
    new_node[:,:,-1] = 1.0 # Set indicator that node is added
    return new_node

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
      # Make new data point and new adjacency connections
      new_node = self._get_new_node(all_nodes)
      new_i = self._get_sparse_adj_indices(all_nodes.size(1))
      new_v = self.kernel(all_nodes, new_node.repeat(1,all_nodes.size(1),1)).squeeze(0)
      all_nodes = torch.cat((all_nodes, new_node),dim=1)
      i = torch.cat((i, new_i),dim=1)
      v = torch.cat((v, new_v, new_v, torch.FloatTensor([1.0]))) # Add 1.0 for new node
      nb_nodes = all_nodes.size(1)
      t = torch.sparse.FloatTensor(i, v, torch.Size([nb_nodes, nb_nodes]))
      adj = torch.sparse.FloatTensor(i, v, torch.Size([nb_nodes, nb_nodes]))
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
    self.create_sparse_graph = PG_Tree(input_dim)
    # self.create_sparse_graph = Rand_Tree(Gaussian(spatial_dims))
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
    '''
    print(emb.size())
    t0 = time.time()
    '''
    # emb, adj = self.create_sparse_graph(emb)
    emb, adj, sum_probs, nb_edges = self.create_sparse_graph(emb.squeeze(0))
    '''
    t1 = time.time()
    '''
    emb = emb.unsqueeze(0)
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
      emb = batch_norm(emb, mask, batch_nb_nodes)
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

def batch_norm(emb, mask, batch_nb_nodes):
  '''
  Normalizes each feature within each sample to mean zero, var 1
  '''
  mean = emb.mean(1)
  emb_centered = emb-mean.unsqueeze(1)

  var = (emb_centered**2).mean(1) + 10**-20
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

def make_ones(size, cuda=False):
  '''
  Make ones accounting for cuda.
  '''
  ones = torch.ones(size)
  return maybe_move_to_cuda(ones,cuda)

def make_zeros(size, cuda=False):
  '''
  Make zeros accounting for cuda.
  '''
  zeros = torch.zeros(size)
  return maybe_move_to_cuda(zeros,cuda)

def maybe_move_to_cuda(tensor, cuda=False):
  '''
  Try to move tensor to cuda if cuda=True.
  '''
  if cuda:
    return tensor.cuda()
  else: 
    return tensor
