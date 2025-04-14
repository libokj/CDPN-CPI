import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp


class AttentionGCN(nn.Module):
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
        x = gmp(x, batch)  # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        return x


class Pocket_BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss(reduce=False)

    def forward(self, pred, label, seq_mask):
        loss_all = self.criterion(pred, label)
        loss = torch.sum(torch.masked_select(loss_all, seq_mask))
        return loss

    def protein_pred_module(self, prot_feature, seq_mask):
        protein_emb = nn.Linear(self.hidden_size1, self.hidden_size1)
        p_feature = F.leaky_relu(protein_emb(prot_feature), 0.1)
        pocket_pred = torch.sigmoid(torch.masked_select(p_feature, seq_mask))

        return pocket_pred
