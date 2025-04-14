from torch import nn
from torch_geometric.nn import GCNConv, global_max_pool


class GCN(nn.Module):
    """
    From `GraphDTA <https://doi.org/10.1093/bioinformatics/btaa921>`_ (Nguyen et al., 2020),
    based on `Graph Convolutional Network <https://arxiv.org/abs/1609.02907>`_ (Kipf and Welling, 2017).
    """
    def __init__(
            self,
            num_features: int,
            out_channels: int,
            dropout: float
    ):
        super().__init__()

        self.conv1 = GCNConv(num_features, num_features)
        self.conv2 = GCNConv(num_features, num_features*2)
        self.conv3 = GCNConv(num_features*2, num_features * 4)
        self.fc_g1 = nn.Linear(num_features*4, 1024)
        self.fc_g2 = nn.Linear(1024, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = global_max_pool(x, batch)  # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        return x
    