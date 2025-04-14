import numpy as np

# Sets of KNOWN characters in SMILES and FASTA sequences
# Use list instead of set to preserve character order
SMILES_VOCAB = ('#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
                '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
                'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
                'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
                'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y')
FASTA_VOCAB = ('A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
               'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z')

# Check uniqueness, create character-index dicts, and add '?' for unknown characters as index 0
assert len(SMILES_VOCAB) == len(set(SMILES_VOCAB)), 'SMILES_CHARSET has duplicate characters.'
SMILES_CHARSET_IDX = {character: index+1 for index, character in enumerate(SMILES_VOCAB)} | {'?': 0}

assert len(FASTA_VOCAB) == len(set(FASTA_VOCAB)), 'FASTA_CHARSET has duplicate characters.'
FASTA_CHARSET_IDX = {character: index+1 for index, character in enumerate(FASTA_VOCAB)} | {'?': 0}


def sequence_to_onehot(sequence: str, charset, max_sequence_length: int, pad_to_max=True):
    assert len(charset) == len(set(charset)), '`charset` contains duplicate characters.'
    charset_idx = {character: index+1 for index, character in enumerate(charset)} | {'?': 0}

    truncated_sequence = sequence[:max_sequence_length]
    sequence_length = max_sequence_length if pad_to_max else len(truncated_sequence)
    onehot = np.zeros((sequence_length, len(charset_idx)), dtype=int)
    for index, character in enumerate(truncated_sequence):
        onehot[index, charset_idx.get(character, 0)] = 1

    return onehot.transpose()


def sequence_to_label(sequence: str, charset, max_sequence_length: int, pad_to_max=True):
    assert len(charset) == len(set(charset)), '`charset` contains duplicate characters.'
    charset_idx = {character: index+1 for index, character in enumerate(charset)} | {'?': 0}

    truncated_sequence = sequence[:max_sequence_length]
    sequence_length = max_sequence_length if pad_to_max else len(truncated_sequence)
    label = np.zeros(sequence_length, dtype=int)
    for index, character in enumerate(truncated_sequence):
        label[index] = charset_idx.get(character, 0)

    return label


def smiles_to_onehot(smiles: str, smiles_charset=SMILES_VOCAB, max_sequence_length: int = 100):  # , in_channels: int = len(SMILES_CHARSET)
    # assert len(SMILES_CHARSET) == len(set(SMILES_CHARSET)), 'SMILES_CHARSET has duplicate characters.'
    # onehot = np.zeros((max_sequence_length, len(SMILES_CHARSET_IDX)))
    # for index, character in enumerate(smiles[:max_sequence_length]):
    #     onehot[index, SMILES_CHARSET_IDX.get(character, 0)] = 1
    # return onehot.transpose()
    return sequence_to_onehot(smiles, smiles_charset, max_sequence_length)


def smiles_to_label(smiles: str, smiles_charset=SMILES_VOCAB, max_sequence_length: int = 100):  # , in_channels: int = len(SMILES_CHARSET)
    # label = np.zeros(max_sequence_length)
    # for index, character in enumerate(smiles[:max_sequence_length]):
    #     label[index] = SMILES_CHARSET_IDX.get(character, 0)
    # return label
    return sequence_to_label(smiles, smiles_charset, max_sequence_length)


def fasta_to_onehot(fasta: str, fasta_charset=FASTA_VOCAB, max_sequence_length: int = 1000):  # in_channels: int = len(FASTA_CHARSET)
    # onehot = np.zeros((max_sequence_length, len(FASTA_CHARSET_IDX)))
    # for index, character in enumerate(fasta[:max_sequence_length]):
    #     onehot[index, FASTA_CHARSET_IDX.get(character, 0)] = 1
    # return onehot.transpose()
    return sequence_to_onehot(fasta, fasta_charset, max_sequence_length)


def fasta_to_label(fasta: str, fasta_charset=FASTA_VOCAB, max_sequence_length: int = 1000):  # in_channels: int = len(FASTA_CHARSET)
    # label = np.zeros(max_sequence_length)
    # for index, character in enumerate(fasta[:max_sequence_length]):
    #     label[index] = FASTA_CHARSET_IDX.get(character, 0)
    # return label
    return sequence_to_label(fasta, fasta_charset, max_sequence_length)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
