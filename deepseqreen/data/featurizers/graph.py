import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.utils import from_smiles
from torch_geometric.data import Data

from deepseqreen.data.featurizers.categorical import one_of_k_encoding_unk, one_of_k_encoding
from deepseqreen.utils import get_logger

log = get_logger(__name__)


def atom_features(atom, explicit_H=False, use_chirality=True):
    """
    Adapted from TransformerCPI 2.0
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridization_type = [Chem.rdchem.HybridizationType.SP,
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3,
                          Chem.rdchem.HybridizationType.SP3D,
                          Chem.rdchem.HybridizationType.SP3D2,
                          'other']  # 6-dim

    # 10+7+2+6+1=26
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridization_type) + [atom.GetIsAromatic()]

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    # 26+5=31
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    # 31+3=34
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array(
        [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
         bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()])


def smiles_to_graph_pyg(smiles):
    """
    Convert SMILES to graph with the default method defined by PyTorch Geometric
    """
    try:
        return from_smiles(smiles)
    except Exception as e:
        log.warning(f"Failed to featurize the following SMILES to graph: {smiles} due to {str(e)}")
        return None


def smiles_to_graph(smiles, atom_features: callable = atom_features):
    """
    Convert SMILES to graph with custom atom_features
    """
    try:
        mol = Chem.MolFromSmiles(smiles)

        features = []
        for atom in mol.GetAtoms():
            feature = atom_features(atom)
            features.append(feature / sum(feature))
        features = np.array(features)

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()

        if len(edges) == 0:
            edge_index = [[0, 0]]
        else:
            edge_index = []
            for e1, e2 in g.edges:
                edge_index.append([e1, e2])

        return Data(x=torch.Tensor(features),
                    edge_index=torch.LongTensor(edge_index).transpose(0, 1))

    except Exception as e:
        log.warning(f"Failed to convert SMILES ({smiles}) to graph due to {str(e)}")
        return None
    # features = []
    # for atom in mol.GetAtoms():
    #     feature = atom_features(atom)
    #     features.append(feature / sum(feature))
    #
    # edge_indices = []
    # for bond in mol.GetBonds():
    #     i = bond.GetBeginAtomIdx()
    #     j = bond.GetEndAtomIdx()
    #     edge_indices += [[i, j], [j, i]]
    #
    # edge_index = torch.tensor(edge_indices)
    # edge_index = edge_index.t().to(torch.long).view(2, -1)
    #
    # if edge_index.numel() > 0:  # Sort indices.
    #     perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
    #     edge_index = edge_index[:, perm]
    #


def smiles_to_mol_features(smiles, num_atom_feat: callable):
    try:
        mol = Chem.MolFromSmiles(smiles)
        num_atom_feat = len(atom_features(mol.GetAtoms()[0]))
        atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
        for atom in mol.GetAtoms():
            atom_feat[atom.GetIdx(), :] = atom_features(atom)
        adj = Chem.GetAdjacencyMatrix(mol)
        adj_mat = np.array(adj)

        return atom_feat, adj_mat

    except Exception as e:
        log.warning(f"Failed to featurize the following SMILES to molecular features: {smiles} due to {str(e)}")
        return None