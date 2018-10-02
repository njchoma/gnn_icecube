import torch

import sparse_with_grad

#####################
#     CONSTANTS     #
#####################
MAX_DEPTH = 2
MAX_NODES = 10

if torch.cuda.is_available():
  t_type = torch.cuda
else:
  t_type = torch

#####################################################
#           CHILD CLASS PG GRAPH CONSTRUCT          #
#####################################################
class MLP_PG(PG_Tree):
  def __init__(self, input_dim, kernel, spatial_dims=None, max_nodes=MAX_DEPTH, max_depth=MAX_DEPTH):
    super(MLP_PG, self).__init__(input_dim, kernel, spatial_dims, max_nodes, max_depth)
    self.subtree_divider = MLP_Divide((input_dim-1)+self.nb_new_features, nb_hidden=1024)

class Rand_Tree(PG_Tree):
  def __init__(self, input_dim, kernel, spatial_dims=None, max_nodes=MAX_DEPTH, max_depth=MAX_DEPTH):
    super(Rand_Tree, self).__init__(input_dim, kernel, spatial_dims, max_nodes, max_depth)
    self.subtree_divider = Random_Divide()


#########################################################
#               ABSTRACT GRAPH CONSTRUCTOR              #
#########################################################
class PG_Tree(nn.Module):
  '''
  Module which recursively divides the input points into two clusters.
  Nodes are added at each division point which connect to all nodes in 
    the respective clusters and to all parent added nodes.
  The output is an augmented point cloud, sparse adjacency matrix, and 
    the sum of probabilities of all paths taken.
  '''
  def __init__(self, input_dim, kernel, spatial_dims=None, max_nodes=MAX_DEPTH, max_depth=MAX_DEPTH):
    '''
    max_nodes is the maximum number of nodes which can belong to a cluster.
    max_depth is the max level of recursion. Prioritized over max_nodes.
    '''
    super(PG_Tree, self).__init__()
    self.max_nodes = max_nodes
    self.max_depth = max_depth
    self.nb_new_features = input_dim-1 # no indicator yet for real/fake nodes
    self.subtree_divider = None
    self.kernel = kernel

  def forward(self, X_in):
    '''
    X_in is assumed to be shape [nb_points, nb_features].
    '''
    # Initialize for DFS
    self._initialize_dfs(X_in)
    torch.set_printoptions(precision=3)
    # Build X features and initial subtree (contains all nodes)
    pg_mask = make_ones(self.nb_nodes, cuda=X_in.is_cuda)
    full_tree = Subtree(t_type.LongTensor(np.arange(self.nb_nodes)),
                        self.max_nodes, t_type.LongTensor([]))
    # DFS
    pg_sum = self.dfs(X_in, [full_tree], pg_mask, depth=0)
    # Prepare augmented nodes
    X_updated = self.gather_nodes(X_in)
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
    '''
    # Use binary edges for now since can't learn in kernel for sparse adj.
    v = make_ones(nb_edges, cuda=X.is_cuda)
    '''
    nodesA = X[idxA].squeeze(1).unsqueeze(0)
    nodesB = X[idxB].squeeze(1).unsqueeze(0)
    v = self.kernel(nodesA, nodesB).squeeze(0)
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
    probs = self.subtree_divider(all_feats)[:,0]
    splits = torch.bernoulli(probs)
    # Sum probabilities for all splits for policy gradient
    try:
      one_class = (splits > 0.5).nonzero()[:,0]
    except:
      one_class = []
    try:
      zero_class = (splits < 0.5).nonzero()[:,0]
    except:
      zero_class = []
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

#################################################
#               SUBTREE DIVIDERS                #
#################################################
class MLP_Divide(nn.Module):
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

class Random_Divide(nn.Module):
  def __init__(self):
    super(Random_Divide, self).__init__()

  def forward(self, X_in):
    return torch.ones(X_in) / 2
