import torch
import torch.nn as nn
from torch.autograd import Variable


class GNN(nn.Module):
  '''
  Graph neural network:
    - Runs through several graph convolution layers of specified type
    - Performs final classification using logistic regression
  '''
  def __init__(self, nb_hidden, nb_layer, input_dim, spatial_dims=None):
    super(GNN, self).__init__()

    self.emb_fc = nn.Linear(input_dim, nb_hidden)
    self.readout_fc = nn.Linear(nb_hidden, 1)
    self.readout_norm = nn.InstanceNorm1d(1)
    self.readout_act = nn.Sigmoid()

  def forward(self, X, mask, batch_nb_nodes):
    batch_size, nb_pts, input_dim  = X.size()

    # Create dummy first adjacency matrix
    adj = Variable(torch.ones(batch_size, nb_pts, nb_pts))
    if X.is_cuda:
      adj = adj.cuda()

    emb = self.emb_fc(X)
    '''
    # Run through layers
    for i, layer in enumerate(self.layers):
      emb, adj = layer(emb, adj, mask, batch_nb_nodes)
    '''

    # Apply final readout and return
    emb = mask_embedding(emb, mask).sum(1)
    emb = self.readout_norm(emb.unsqueeze(1)).squeeze(1)
    emb = self.readout_fc(emb)
    emb = self.readout_act(emb).squeeze(1)
    return emb


###########################
#     MODEL UTILITIES     #
###########################
def mask_embedding(emb, mask):
  nb_features = emb.size()[2]
  return torch.mul(emb, mask[:,:,0:1].repeat(1,1,nb_features))
