import torch
from torch import nn


class GraphConv(nn.Module):
    def __init__(self, inp, out):
        super(GraphConv, self).__init__()

        self.inp = inp
        self.out = out
        self.weight = nn.Parameter(torch.FloatTensor(inp, out))

    def forward(self, x, adj):
        """
        Calculates A' . X . W, where A' is normalized A -> D^-0.5 . A . D^-0.5
        :param x: input, or hidden layer
        :param adj: normalized adjacency matrix
        :return: adj . x . weight
        """
        output = torch.mm(x, self.weight)
        return torch.mm(adj, output)    # can use sparse mm since adj is a sparse matrix
