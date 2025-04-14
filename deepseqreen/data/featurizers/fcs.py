from importlib import resources

import numpy as np
import pandas as pd
from subword_nmt.apply_bpe import BPE
import codecs

vocab_path = resources.files('deepseqreen').parent.joinpath('resources/vocabs/ESPF/protein_codes_uniprot.txt')
bpe_codes_protein = codecs.open(vocab_path)
protein_bpe = BPE(bpe_codes_protein, merges=-1, separator='')

sub_csv_path = resources.files('deepseqreen').parent.joinpath('resources/vocabs/ESPF/subword_units_map_uniprot.csv')
sub_csv = pd.read_csv(sub_csv_path)
idx2word_protein = sub_csv['index'].values
words2idx_protein = dict(zip(idx2word_protein, range(0, len(idx2word_protein))))

vocab_path = resources.files('deepseqreen').parent.joinpath('resources/vocabs/ESPF/drug_codes_chembl.txt')
bpe_codes_drug = codecs.open(vocab_path)
drug_bpe = BPE(bpe_codes_drug, merges=-1, separator='')

sub_csv_path = resources.files('deepseqreen').parent.joinpath('resources/vocabs/ESPF/subword_units_map_chembl.csv')
sub_csv = pd.read_csv(sub_csv_path)
idx2word_drug = sub_csv['index'].values
words2idx_drug = dict(zip(idx2word_drug, range(0, len(idx2word_drug))))


def protein_to_embedding(x, max_sequence_length):
    max_p = max_sequence_length
    t1 = protein_bpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_protein[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        # print(x)

    l = len(i1)

    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p

    return i, np.asarray(input_mask)


def drug_to_embedding(x, max_sequence_length):
    max_d = max_sequence_length
    t1 = drug_bpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_drug[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        # print(x)

    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)
