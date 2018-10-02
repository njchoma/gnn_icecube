import torch

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
