import torch
import torch.nn as nn

class GNN(nn.Module):
  def __init__(self, input_dim, nb_hidden, nb_layers):
    super(GNN, self).__init__()
    self.fc = nn.Linear(10, 10)
