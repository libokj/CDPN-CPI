"""
Mainly adapted from MolMap:
https://github.com/shenwanxiang/bidd-molmap/tree/master/molmap/feature/fingerprint
"""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint

from deepseqreen import get_logger

log = get_logger(__name__)


def smiles_to_erg(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        features = np.array(GetErGFingerprint(mol), dtype=bool)
        return features
    except Exception as e:
        log.warning(f"Failed to convert SMILES ({smiles}) to ErGFP due to {str(e)}")
        return None


def smiles_to_morgan(smiles, radius=2, n_bits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except Exception as e:
        log.warning(f"Failed to convert SMILES ({smiles}) to ErGFP due to {str(e)}")
        return None


def smiles_to_daylight(smiles):
    try:
        NumFinger = 2048
        mol = Chem.MolFromSmiles(smiles)
        bv = FingerprintMols.FingerprintMol(mol)
        temp = tuple(bv.GetOnBits())
        features = np.zeros((NumFinger,))
        features[np.array(temp)] = 1
    except:
        print(f'RDKit could not find this SMILES: {smiles} convert to all 0 features')
        features = np.zeros((2048,))
    return features.astype(int)
