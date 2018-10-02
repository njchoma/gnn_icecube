import torch
import torch.nn as nn

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
    self.sigma = nn.Parameter(torch.FloatTensor([1.0]))
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

class MLP(nn.Module):
    def __init__(self, input_dim, nb_hidden, apply_sigmoid=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim*2, nb_hidden)
        self.act1 = nn.SELU()
        self.fc2 = nn.Linear(nb_hidden, 1)
        self.apply_sigmoid = apply_sigmoid
        if apply_sigmoid:
            self.act2 = nn.Sigmoid()

    def forward(self, emb_in_0, emb_in_1):
        X = torch.cat((emb_in_0, emb_in_1), dim=2)
        h = self.act1(self.fc1(X))
        o = self.fc2(h)
        if self.apply_sigmoid:
            o = self.act2(o)
        return o.squeeze(2)
