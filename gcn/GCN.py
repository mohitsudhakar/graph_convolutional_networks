from torch import nn
import torch.nn.functional as F
from gcn.GraphConv import GraphConv


class GCN(nn.Module):
    def __init__(self, num_features, hidden_layers, num_labels, dropout):
        """
        :param num_layers: Number of layers (hidden + output)
        :param num_features: input layer size
        :param hidden_layers: list of hidden layer size
        :param num_labels: output layer
        :param dropout: dropout
        """
        super(GCN, self).__init__()

        # self.num_layers = len(hidden_layers) + 1
        # if self.num_layers == 1:
        #     self.graph_convolutions = [GraphConv(num_features, num_labels)]
        # else:
        #     self.graph_convolutions = [GraphConv(num_features, hidden_layers[0])]
        #     self.graph_convolutions.append(nn.ReLU())
        #     self.graph_convolutions.append(nn.Dropout(dropout))
        #     # num_layers = 2, hidden = [3],
        #     if len(hidden_layers) > 1:
        #         for l in range(0, len(hidden_layers) - 1):
        #             gc = GraphConv(hidden_layers[l], hidden_layers[l+1])
        #             self.graph_convolutions.append(gc)
        #             self.graph_convolutions.append(nn.ReLU())
        #             self.graph_convolutions.append(nn.Dropout(dropout))
        #
        #     self.graph_convolutions.append(GraphConv(hidden_layers[-1], num_labels))
        #     self.graph_convolutions.append(nn.LogSoftmax())
        #
        # self.gcn = nn.Sequential(*self.graph_convolutions)

        self.gc1 = GraphConv(num_features, hidden_layers[0])
        self.gc2 = GraphConv(hidden_layers[0], num_labels)
        self.dropout = dropout

    def forward(self, x, adj):

        # x = self.gcn(x, adj)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
