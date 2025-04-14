from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp


class GAT(nn.Module):
    r"""
    From `GraphDTA <https://doi.org/10.1093/bioinformatics/btaa921>`_ (Nguyen et al., 2020),
    based on `Graph Attention Network <https://arxiv.org/abs/1710.10903>`_ (Veličković et al., 2018).
    """
    def __init__(
        self,
        num_features: int,
        out_channels: int,
        dropout: float
    ):
        super().__init__()

        self.dropout = dropout
        self.gcn1 = GATConv(num_features, num_features, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features * 10, out_channels, dropout=dropout)
        self.fc_g1 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, data):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)  # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        return x
