from torch import cat, nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GATGCN(nn.Module):
    r"""
    From `GraphDTA <https://doi.org/10.1093/bioinformatics/btaa921>`_ (Nguyen et al., 2020),
    based on `Graph Attention Network <https://arxiv.org/abs/1710.10903>`_ (Veličković et al., 2018)
    and `Graph Convolutional Network <https://arxiv.org/abs/1609.02907>`_ (Kipf and Welling, 2017).
    """
    def __init__(
            self,
            num_features: int,
            out_channels: int,
            dropout: float
    ):
        super().__init__()

        self.conv1 = GATConv(num_features, num_features, heads=10)
        self.conv2 = GCNConv(num_features*10, num_features*10)
        self.fc_g1 = nn.Linear(num_features*10*2, 1500)
        self.fc_g2 = nn.Linear(1500, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print('x shape = ', x.shape)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        return x
