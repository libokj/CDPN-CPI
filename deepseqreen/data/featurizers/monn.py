import numpy as np
from rdkit.Chem import MolFromSmiles

from deepseqreen.data.featurizers.categorical import FASTA_VOCAB, fasta_to_label
from deepseqreen.data.featurizers.graph import atom_features, bond_features


def get_mask(arr):
    a = np.zeros(1, len(arr))
    a[1, :arr.shape[0]] = 1
    return a


def add_index(input_array, ebd_size):
    batch_size, n_vertex, n_nbs = np.shape(input_array)
    add_idx = np.array(range(0, ebd_size * batch_size, ebd_size) * (n_nbs * n_vertex))
    add_idx = np.transpose(add_idx.reshape(-1, batch_size))
    add_idx = add_idx.reshape(-1)
    new_array = input_array.reshape(-1) + add_idx
    return new_array


# TODO fix padding and masking
def drug_featurizer(smiles, max_neighbors=6):
    mol = MolFromSmiles(smiles)

    # convert molecule to GNN input
    n_atoms = mol.GetNumAtoms()
    assert mol.GetNumBonds() >= 0

    n_bonds = max(mol.GetNumBonds(), 1)
    feat_atoms = np.zeros((n_atoms,))  # atom feature ID
    feat_bonds = np.zeros((n_bonds,))  # bond feature ID
    atom_adj = np.zeros((n_atoms, max_neighbors))
    bond_adj = np.zeros((n_atoms, max_neighbors))
    n_neighbors = np.zeros((n_atoms,))
    neighbor_mask = np.zeros((n_atoms, max_neighbors))

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        feat_atoms[idx] = atom_features(atom)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        idx = bond.GetIdx()
        feat_bonds[idx] = bond_features(bond)
        try:
            atom_adj[a1, n_neighbors[a1]] = a2
            atom_adj[a2, n_neighbors[a2]] = a1
        except:
            return [], [], [], [], []
        bond_adj[a1, n_neighbors[a1]] = idx
        bond_adj[a2, n_neighbors[a2]] = idx
        n_neighbors[a1] += 1
        n_neighbors[a2] += 1

    for i in range(len(n_neighbors)):
        neighbor_mask[i, :n_neighbors[i]] = 1

    vertex_mask = get_mask(feat_atoms)
    # vertex = pack_1d(feat_atoms)
    # edge = pack_1d(feat_bonds)
    # atom_adj = pack_2d(atom_adj)
    # bond_adj = pack_2d(bond_adj)
    # nbs_mask = pack_2d(n_neighbors_mat)

    atom_adj = add_index(atom_adj, np.shape(atom_adj)[1])
    bond_adj = add_index(bond_adj, np.shape(feat_bonds)[1])

    return vertex_mask, feat_atoms, feat_bonds, atom_adj, bond_adj, neighbor_mask


# TODO WIP the pairwise_label matrix probably should be generated beforehand and stored as an extra label in the dataset
def get_pairwise_label(pdbid, interaction_dict, mol):
    if pdbid in interaction_dict:
        sdf_element = np.array([atom.GetSymbol().upper() for atom in mol.GetAtoms()])
        atom_element = np.array(interaction_dict[pdbid]['atom_element'], dtype=str)
        atom_name_list = np.array(interaction_dict[pdbid]['atom_name'], dtype=str)
        atom_interact = np.array(interaction_dict[pdbid]['atom_interact'], dtype=int)
        nonH_position = np.where(atom_element != 'H')[0]
        assert sum(atom_element[nonH_position] != sdf_element) == 0

        atom_name_list = atom_name_list[nonH_position].tolist()
        pairwise_mat = np.zeros((len(nonH_position), len(interaction_dict[pdbid]['uniprot_seq'])), dtype=np.int32)
        for atom_name, bond_type in interaction_dict[pdbid]['atom_bond_type']:
            atom_idx = atom_name_list.index(str(atom_name))
            assert atom_idx < len(nonH_position)

            seq_idx_list = []
            for seq_idx, bond_type_seq in interaction_dict[pdbid]['residue_bond_type']:
                if bond_type == bond_type_seq:
                    seq_idx_list.append(seq_idx)
                    pairwise_mat[atom_idx, seq_idx] = 1
        if len(np.where(pairwise_mat != 0)[0]) != 0:
            pairwise_mask = True
            return True, pairwise_mat
    return False, np.zeros((1, 1))


def protein_featurizer(fasta):
    sequence = fasta_to_label(fasta)
    # pad proteins and make masks
    seq_mask = get_mask(sequence)

    return seq_mask, sequence
