import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_max_pool as gmp
from torch_geometric.utils import to_dense_batch

from deepseqreen.data.featurizers.categorical import one_of_k_encoding_unk, one_of_k_encoding


class PMFCPI(nn.Module):
    def __init__(
            self,
            pretrain,
            emb_size=768, hidden_size=128, num_features_mol=78, dropout=0.2,  # max_length=1500
    ):
        super().__init__()
        self.pretrain = pretrain
        # self.max_length = max_length
        self.emb_size = emb_size
        # compounds network
        self.mol_sage = SAGE(num_features_mol, hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)

        # proteins network
        self.prot_rnn = nn.LSTM(self.emb_size, hidden_size, 1)
        self.relu = nn.LeakyReLU()
        # combined layers
        self.prot_comp_mix = nn.Sequential(
            nn.Linear(129 * 129, 1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        self.fc = nn.Sequential(
            nn.Linear(1282, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, data_mol, data_pro):
        pro_token_ids, pro_seq_lengths = data_pro, data_pro.lengths
        with torch.no_grad():
            data_pro = self.pretrain(pro_token_ids)[0]

        pro_seq_lengths = torch.sort(pro_seq_lengths, descending=True)[::-1][1]
        pro_idx_sort = torch.argsort(-pro_seq_lengths)
        pro_idx_unsort = torch.argsort(pro_idx_sort)
        data_pro = data_pro.index_select(0, pro_idx_sort)
        xt = nn.utils.rnn.pack_padded_sequence(data_pro, pro_seq_lengths.cpu(), batch_first=True)
        xt, _ = self.prot_rnn(xt)
        xt = nn.utils.rnn.pad_packed_sequence(
            xt,
            batch_first=True,
            # total_length=self.max_length
        )[0]
        xt = xt.index_select(0, pro_idx_unsort)
        xt = xt.mean(1)

        # compound network
        x = self.mol_sage(data_mol)

        # kronecker product
        prot_out = torch.cat((xt, torch.ones(xt.shape[0], 1, device=xt.device)), dim=1)
        comp_out = torch.cat((x, torch.ones(x.shape[0], 1, device=x.device)), dim=1)
        output = torch.bmm(prot_out.unsqueeze(2), comp_out.unsqueeze(1)).flatten(start_dim=1)
        output = self.dropout(output)
        output = self.prot_comp_mix(output)
        output = torch.cat((output, prot_out, comp_out), 1)
        output = self.fc(output)

        return output


class SAGE(nn.Module):
    def __init__(
            self,
            num_features=78, out_channels=128, dropout=0.2,
            pool=True, add_super_node=False
     ):
        super().__init__()
        self.mol_conv1 = SAGEConv(num_features, num_features * 2, 'mean')
        self.mol_conv2_f = SAGEConv(num_features * 2, num_features * 2, 'mean')
        self.mol_conv3_f = SAGEConv(num_features * 2, num_features * 4, 'mean')
        self.mol_fc_g1 = nn.Linear(num_features * 4, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.pool = pool
        self.add_super_node = add_super_node

    def forward(self, mol):
        x, edge_index, batch = mol.x, mol.edge_index, mol.batch

        if self.add_super_node:
            # Add a virtual node with feature 0 and connect it to all nodes
            num_nodes = x.size(0)
            virtual_node_feature = torch.zeros((1, x.size(1)), device=x.device)
            x = torch.cat([virtual_node_feature, x], dim=0)
            edge_index = edge_index + 1
            virtual_node_edge_index = torch.stack([
                torch.zeros(num_nodes, dtype=torch.long, device=x.device),
                torch.arange(1, num_nodes + 1, device=x.device)
            ], dim=0)
            edge_index = torch.cat([
                edge_index,
                virtual_node_edge_index,
                virtual_node_edge_index.flip(0)
            ], dim=1)
            batch = torch.cat([torch.zeros(1, dtype=batch.dtype, device=batch.device), batch])

        x = self.mol_conv1(x, edge_index)
        x = self.relu(x)
        x = self.mol_conv2_f(x, edge_index)
        x = self.relu(x)
        x = self.mol_conv3_f(x, edge_index)
        x = self.relu(x)

        if self.pool:
            if self.add_super_node:
                # If adding a super node and pooling, use only the super node representation
                x, mask = to_dense_batch(x, batch)
                # x = x[:, 0, :]  # Use only the super node (index 0) representation
            else:
                # global max pooling
                x = gmp(x, batch)

        x = self.mol_fc_g1(x)
        x = self.dropout(x)

        if not self.pool:
            x, mask = to_dense_batch(x, batch)
            x.mask = ~mask

        if self.add_super_node:
            x.add_super_node = True

        return x


class LSTMEncoder(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden_size, 1)

    def forward(self, v):
        lengths = torch.sort(v.lengths, descending=True)[::-1][1]
        idx_sort = torch.argsort(-lengths)
        idx_unsort = torch.argsort(idx_sort)
        v = v.index_select(0, idx_sort)
        v = nn.utils.rnn.pack_padded_sequence(v, lengths.cpu(), batch_first=True)
        v, _ = self.lstm(v)
        v = nn.utils.rnn.pad_packed_sequence(
            v,
            batch_first=True,
            # total_length=self.max_length
        )[0]
        v = v.index_select(0, idx_unsort)
        v = v.mean(1)
        return v


def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al',
        'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
        'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'X'
    ]) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()

    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))

    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])

    return Data(
        x=torch.Tensor(np.array(features)),
        edge_index=torch.LongTensor(np.array(edge_index)).transpose(1, 0)
    )
