import torch
import torch.nn as nn
import numpy as np

from deepseqreen.data.featurizers.categorical import one_of_k_encoding, one_of_k_encoding_unk


class GraphDTA(nn.Module):
    """
    From GraphDTA (Nguyen et al., 2020; https://doi.org/10.1093/bioinformatics/btaa921).
    """
    def __init__(
            self,
            gnn: nn.Module,
            num_features_protein: int,
            n_filters: int,
            embed_dim: int,
            output_dim: int,
            dropout: float
    ):
        super().__init__()
        self.gnn = gnn

        # protein sequence encoder (1d conv)
        self.embedding_xt = nn.Embedding(num_features_protein, embed_dim)
        self.conv_xt = nn.LazyConv1d(out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32 * 121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # protein input feedforward
    def conv_forward_xt(self, v_p):
        v_p = self.embedding_xt(v_p.long())
        v_p = self.conv_xt(v_p)
        # flatten
        v_p = v_p.view(-1, 32 * 121)
        v_p = self.fc1_xt(v_p)
        return v_p

    def forward(self, v_d, v_p):
        v_d = self.gnn(v_d)
        v_p = self.conv_forward_xt(v_p)

        # concat
        v_f = torch.cat((v_d, v_p), 1)
        # dense layers
        v_f = self.fc1(v_f)
        v_f = self.relu(v_f)
        v_f = self.dropout(v_f)
        v_f = self.fc2(v_f)
        v_f = self.relu(v_f)
        v_f = self.dropout(v_f)
        # v_f = self.out(v_f)
        return v_f

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                    ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                     'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb',
                     'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu',
                     'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])
