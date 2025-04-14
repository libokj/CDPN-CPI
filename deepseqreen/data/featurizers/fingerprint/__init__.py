from typing import Literal

from .atompairs import GetAtomPairFPs
from .avalonfp import GetAvalonFPs
from .rdkitfp import GetRDkitFPs
from .morganfp import GetMorganFPs
from .estatefp import GetEstateFPs
from .maccskeys import GetMACCSFPs
from .pharmErGfp import GetPharmacoErGFPs
from .pharmPointfp import GetPharmacoPFPs
from .pubchemfp import GetPubChemFPs
from .torsions import GetTorsionFPs
from .mhfp6 import GetMHFP6
# from .map4 import GetMAP4
from rdkit import Chem

from deepseqreen import get_logger

log = get_logger(__name__)

FP_MAP = {
    'MorganFP': GetMorganFPs,
    'RDkitFP': GetRDkitFPs,
    'AtomPairFP': GetAtomPairFPs,
    'TorsionFP': GetTorsionFPs,
    'AvalonFP': GetAvalonFPs,
    'EstateFP': GetEstateFPs,
    'MACCSFP': GetMACCSFPs,
    'PharmacoErGFP': GetPharmacoErGFPs,
    'PharmacoPFP': GetPharmacoPFPs,
    'PubChemFP': GetPubChemFPs,
    'MHFP6': GetMHFP6,
    # 'MAP4': GetMAP4,
}


def smiles_to_fingerprint(smiles, fingerprint: Literal[tuple(FP_MAP.keys())], **kwargs):
    func = FP_MAP[fingerprint]
    try:
        mol = Chem.MolFromSmiles(smiles)
        arr = func(mol, **kwargs)
        return arr
    except Exception as e:
        log.warning(f"Failed to convert SMILES ({smiles}) to {fingerprint} due to {str(e)}")
        return None
