import ast
from collections import defaultdict
import re
from functools import partial, cache
from numbers import Number
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, Union, Literal

from lightning import LightningDataModule
import numpy as np
from omegaconf import ListConfig, DictConfig
import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem
#import swifter
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

from deepseqreen.data.utils import label_transform, collate_fn, SafeBatchSampler, convert_none_like_str_to_none, worker_init_fn
from deepseqreen.data.featurizers.loader import BaseFeatureLoader
from deepseqreen.utils import get_logger

log = get_logger(__name__)
pandarallel.initialize()

SMILES_PAT = r"[^A-Za-z0-9=#:+\-\[\]<>()/\\@%,.*]"
FASTA_PAT = r"[^A-Z*\-]"


def ensure_cached(func, keys=None):
    if keys is None:
        keys = set()
    if hasattr(func, 'cache_info'):
        return func

    elif hasattr(func, '__call__'):
        if isinstance(func, BaseFeatureLoader):
            if hasattr(func, 'in_memory') and getattr(func, 'in_memory') == True:
                if hasattr(func, 'feature_map') and len(getattr(func, 'feature_map')) == 0:
                    if hasattr(func, 'load_feature_map'):
                        func.load_feature_map(keys)
                        return func

        func.__call__ = cache(func.__call__)
        return func


def cache_clear(func):
    if hasattr(func, 'cache_clear'):
        func.cache_clear()
    elif hasattr(func, '__call__') and hasattr(func.__call__, 'cache_clear'):
        func.__call__.cache_clear()
    elif hasattr(func, 'close'):
        func.close()


@cache
def validate_seq_str(seq, regex):
    if seq:
        err_charset = set(re.findall(regex, seq))
        if not err_charset:
            return None
        else:
            return ', '.join(err_charset)
    else:
        return 'Empty string'


# TODO: save a list of corrupted records
@cache
def rdkit_canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        smiles = Chem.MolToSmiles(mol)
    else:
        log.warning(f'Failed to canonicalize SMILES using RDKit. Returning original SMILES: {smiles}')
    return smiles


def safe_literal_eval(x):
    if pd.isna(x):
        return None
    try:
        return np.array(ast.literal_eval(x))
    except (ValueError, SyntaxError):
        return x


def read_csv(filepath, extra_keys):
    log.info(f"Processing data file: {filepath}")

    df = pd.read_csv(
        filepath,
        engine='python', header=0,
        usecols=lambda x: x in ['X1', 'ID1', 'X2', 'ID2', 'Y', 'U'] + list(extra_keys),
        dtype={
            'X1': 'str', 'ID1': 'str',
            'X2': 'str', 'ID2': 'str',
            'Y': 'float32', 'U': 'str',
        }
    )

    entity_cols = ['X1', 'X2']
    # Forward-fill all non-label columns
    df.loc[:, df.columns != 'Y'] = df.loc[:, df.columns != 'Y'].ffill(axis=0)

    if 'X1' in df.columns:
        log.info("Validating SMILES (`X1`)...")
        assert not df['X1'].isna().any() is False, f'`X1` contains NA'
        df['X1_ERR'] = df['X1'].apply(validate_seq_str, regex=SMILES_PAT)
        validate_seq_str.cache_clear()

        if not df['X1_ERR'].isna().all():
            raise Exception(f"Encountered invalid SMILES:\n{df[~df['X1_ERR'].isna()][['X1', 'X1_ERR']]}")
        log.info("Canonicalizing SMILES (`X1`)...")
        df['X1^'] = df['X1'].apply(rdkit_canonicalize)
        rdkit_canonicalize.cache_clear()

    if 'X2' in df.columns:
        log.info("Validating FASTA (`X2`)...")
        assert not df['X2'].isna().any() is False, f'`X2` contains NA'
        df['X2'] = df['X2'].str.upper()
        df['X2_ERR'] = df['X2'].apply(validate_seq_str, regex=FASTA_PAT)
        validate_seq_str.cache_clear()

        if not df['X2_ERR'].isna().all():
            raise Exception(f"Encountered invalid FASTA:\n{df[~df['X2_ERR'].isna()][['X2', 'X2_ERR']]}")

    for key in extra_keys:
        assert key in df.columns, f"Extra key '{key}' not found in dataset file ({filepath})."
        if df[key].dtype in ['object', 'str']:
            try:
                df[key] = df[key].apply(safe_literal_eval)
            except:
                pass

    # Fill NAs in string cols with an empty string to prevent wrong type inference in pytorch collator
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')

    return df


def process_data_file(data_file, data_dir, extra_keys):
    if isinstance(data_file, list | ListConfig):
        processed_data = []
        for data_file in list(data_file):
            data_path = Path(data_file)
            if not data_path.is_absolute():
                data_path = Path(data_dir, data_path)
            processed_data.append(read_csv(data_path, extra_keys))

    if isinstance(data_file, dict | DictConfig):
        processed_data = {}
        for key, data_file in dict(data_file).items():
            data_path = Path(data_file)
            if not data_path.is_absolute():
                data_path = Path(data_dir, data_path)
            processed_data[key] = read_csv(data_path, extra_keys)

    else:
        data_path = Path(data_file)
        if not data_path.is_absolute():
            data_path = Path(data_dir, data_path)
        processed_data = read_csv(data_path, extra_keys)

    return processed_data


class DTIDataset(Dataset):
    optional_keys = ['ID1', 'ID2', 'ID^', 'Y', 'weight']

    def __init__(
            self,
            task: Literal['regression', 'binary', 'multiclass'],
            num_classes: Optional[int],
            df: pd.DataFrame,
            drug_featurizer: callable,
            protein_featurizer: callable,
            thresholds: Optional[Union[Number, Iterable[Number]]] = None,
            discard_intermediate: Optional[bool] = False,
            query: Optional[str] = 'X2',
            preprocess: Optional[Union[callable, bool]] = None,
            extra_keys: Optional[Iterable[str]] = (),
    ):
        # if 'ID1' in df:
        #     self.x1_to_id1 = dict(zip(df['X1'], df['ID1']))
        # if 'ID2' in df:
        #     self.x2_to_id2 = dict(zip(df['X2'], df['ID2']))
        #     self.id2_to_indexes = dict(zip(df['ID2'], range(len(df['ID2']))))
        # self.x2_to_indexes = dict(zip(df['X2'], range(len(df['X2']))))

        # if 'Y' in df.columns and df['Y'].notnull().all():
        optional_keys = [key for key in self.optional_keys if key in df.columns]
        self.extra_keys = list(set(optional_keys + list(extra_keys)))

        if 'Y' in df:
            log.info(f"Validating labels (`Y`)...")
            # TODO: check sklearn.utils.multiclass.check_classification_targets
            match task:
                case 'regression':
                    assert all(df['Y'].apply(lambda x: isinstance(x, Number))), \
                        f"""`Y` must be numeric for `regression` task,
                        but it has {set(df['Y'].apply(type))}."""

                case 'binary':
                    if df['Y'].isin([0, 1]).all():
                        assert not thresholds, \
                            f"""`Y` is already 0 or 1 for `binary` (classification) `task`,
                            but still got `thresholds` ({thresholds}).
                            Double check your choices of `task` and `thresholds`, and records in the `Y` column."""
                    else:
                        assert thresholds, \
                            f"""`Y` must be 0 or 1 for `binary` (classification) `task`,
                            but it has {pd.unique(df['Y'])}.
                            You may set `thresholds` to discretize continuous labels."""

                case 'multiclass':
                    assert num_classes >= 3, f'`num_classes` for `task=multiclass` must be at least 3.'

                    if all(df['Y'].apply(lambda x: x.is_integer() and x >= 0)):
                        assert not thresholds, \
                            f"""`Y` is already non-negative integers for 
                            `multiclass` (classification) `task`, but still got `thresholds` ({thresholds}).
                            Double check your choice of `task`, `thresholds` and records in the `Y` column."""
                    else:
                        assert thresholds, \
                            f"""`Y` must be non-negative integers for
                            `multiclass` (classification) 'task',but it has {pd.unique(df['Y'])}.
                            You must set `thresholds` to discretize continuous labels."""  # TODO print err idx instead

            if 'U' in df.columns:
                units = df['U']
            else:
                units = None
                log.warning("Units ('U') not in the data table. "
                            "Assuming all labels to be discrete or in p-scale (-log10[M]).")

            # Transform labels
            df['Y'] = label_transform(labels=df['Y'], units=units, thresholds=thresholds,
                                      discard_intermediate=discard_intermediate)

            # Filter out rows with a NaN in Y (missing values)
            df.dropna(subset=['Y'], inplace=True)

            match task:
                case 'regression':
                    df['Y'] = df['Y'].astype('float32')
                    assert all(df['Y'].apply(lambda x: isinstance(x, Number))), \
                        f"""`Y` must be numeric for `regression` task,
                        but after transformation it still has {set(df['Y'].apply(type))}.
                        Double check your choices of `task` and `thresholds` and records in the `Y` and `U` columns."""
                    # TODO print err idx instead
                case 'binary':
                    df['Y'] = df['Y'].astype('int')
                    assert all(df['Y'].isin([0, 1])), \
                        f"""`Y` must be 0 or 1 for `task=binary`, "
                        but after transformation it still has {pd.unique(df['Y'])}.
                        Double check your choices of `task` and `thresholds` and records in the `Y` and `U` columns."""
                    # TODO print err idx instead
                case 'multiclass':
                    df['Y'] = df['Y'].astype('int')
                    assert all(df['Y'].apply(lambda x: x.is_integer() and x >= 0)), \
                        f"""Y must be non-negative integers for `task=multiclass`
                        but after transformation it still has {pd.unique(df['Y'])}.
                        Double check your choices of `task` and `thresholds` and records in the `Y` and `U` columns."""
                    # TODO print err idx instead
                    target_n_unique = df['Y'].nunique()
                    assert target_n_unique == num_classes, \
                        f"""You have set `num_classes` for `task=multiclass` to {num_classes},
                        but after transformation Y still has {target_n_unique} unique labels.
                        Double check your choices of `task` and `thresholds` and records in the `Y` and `U` columns."""

        # FASTA/SMILES indices as query for retrieval metrics like enrichment factor and hit rate
        if query:
            df['ID^'] = LabelEncoder().fit_transform(df[query])

        self.df = df
        self.extra_keys = list(set([
            key for key in (self.optional_keys + list(extra_keys)) if key in df.columns
        ]))

        if drug_featurizer is not None:
            self.drug_featurizer = drug_featurizer
            if preprocess is True:
                log.info('Featurizing SMILES...')
                self.df['X1^'].apply(drug_featurizer)
        else:
            self.drug_featurizer = (lambda x: x)

        if protein_featurizer is not None:
            self.protein_featurizer = protein_featurizer
            if preprocess is True:
                log.info('Featurizing FASTA...')
                self.df['X2'].apply(protein_featurizer)
        else:
            self.protein_featurizer = (lambda x: x)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        sample = self.df.iloc[i]
        sample_dict = {
            'N': i,
            'X1': sample['X1'],
            'X1^': self.drug_featurizer(sample['X1^']),
            # 'ID1': sample.get('ID1'),
            'X2': sample['X2'],
            'X2^': self.protein_featurizer(sample['X2']),
            # 'ID2': sample.get('ID2'),
            # 'Y': sample.get('Y'),
            # 'ID^': sample.get('ID^'),
        } | {
            key: sample[key] for key in self.extra_keys
        }
        return sample_dict


class DTIDataModule(LightningDataModule):
    """
    DTI DataModule

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc.
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc.
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            task: Literal['regression', 'binary', 'multiclass'],
            num_classes: Optional[int],
            batch_size: int,
            # train: bool,
            drug_featurizer: callable,
            protein_featurizer: callable,
            collator: callable = collate_fn,
            data_dir: str = "data/",
            data_file: Optional[str] = None,
            train_val_test_split: Optional[Union[Iterable[Number | str]]] = None,
            split: Optional[callable] = None,
            thresholds: Optional[Union[Number, Iterable[Number]]] = None,
            discard_intermediate: Optional[bool] = False,
            query: Optional[str] = 'X2',
            num_workers: int = 0,
            pin_memory: bool = True,
            preprocess: Optional[Union[callable, bool]] = None,
            extra_keys: Optional[Iterable[str]] = (),
    ):
        super().__init__()

        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        self.predict_data: Optional[Dataset] = None

        self.split = split
        self.collator = collator

        self.featurizers = {'drug': drug_featurizer, 'protein': protein_featurizer}
        self.extra_keys = [key for key in extra_keys if key != 'label']

        self.dataset = partial(
            DTIDataset,
            task=task,
            num_classes=num_classes,
            thresholds=thresholds,
            discard_intermediate=discard_intermediate,
            query=query,
            preprocess=preprocess,
            extra_keys=self.extra_keys
        )
        
        self.data_dir = data_dir
        self.data_file = data_file
        self.train_val_test_split = train_val_test_split

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def prepare_data(self):
        """
        Download data if needed.
        Do not use it to assign state (e.g., self.x = x).
        """

    def setup(self, stage: Optional[str] = None, encoding: str = None):
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute data splitting twice.
        """
        data_list = ['train_data', 'val_data', 'test_data', 'predict_data']
        # load and split datasets only if not loaded in initialization
        if stage == 'predict' and self.predict_data is None:
            data_path_list = []

            if not isinstance(self.data_file, list | ListConfig):
                self.data_file = [self.data_file]

            for data_file in list(self.data_file):
                data_path = Path(data_file)
                if not data_path.is_absolute():
                    data_path = Path(self.data_dir, data_path)
                data_path_list.append(data_path)

            if len(data_path_list) == 1:
                df = read_csv(data_path_list[0], self.extra_keys)

            elif len(data_path_list) == 2:
                df = read_csv(data_path_list[0], self.extra_keys).merge(
                    read_csv(data_path_list[1], self.extra_keys), how='cross'
                ).drop_duplicates(subset=['X1', 'X2'])

            else:
                raise ValueError('For prediction, you must specify either 1 file for interaction pairs, '
                                 'or 2 files for screening a set of entities against another library of entities.')

            self.predict_data = df

        elif stage == 'test' and self.test_data is None:
            if self.get('data_file') and not self.get('train_val_test_split'):
                self.test_data = process_data_file(self.data_file, self.data_dir, self.extra_keys)
            elif not self.get('data_file') and self.get('train_val_test_split'):
                if len(self.train_val_test_split) == 3:
                    self.test_data = process_data_file(list(self.train_val_test_split)[2], self.data_dir, self.extra_keys)
                else:
                    raise ValueError('Length of `train_val_test_split` must be 3. For testing, the third element (2) in it should be a filepath or a list/dict of filepaths.')
            else:
                raise ValueError('For testing, you must specify a filepath or a list/dict of filepaths with `data.data_file` or the third element (2) in `data.train_val_test_split`')

        elif stage =='validate':
            return

        elif all([getattr(self, data) is None for data in data_list]):
            if self.train_val_test_split:
                if len(self.train_val_test_split) != 3:
                    raise ValueError('Length of `train_val_test_split` must be 3. '
                                     'Set the second element to `null` for training without validation. '
                                     'Set the third element to `null` for training without testing.')

                self.train_val_test_split = convert_none_like_str_to_none(self.train_val_test_split)

                self.train_data = self.train_val_test_split[0]
                self.val_data = self.train_val_test_split[1]
                self.test_data = self.train_val_test_split[2]
                if all([self.data_file, self.split]):
                    if all(isinstance(split, Number) or split is None for split in self.train_val_test_split):
                        df = read_csv(Path(self.data_dir, self.data_file), self.extra_keys)
                        log.info(f'Applying train/val/test split ({self.train_val_test_split})...')
                        split_data = self.split(
                            dataset=df,
                            lengths=[split for split in self.train_val_test_split if split is not None]
                        )
                        for dataset in ['train_data', 'val_data', 'test_data']:
                            if getattr(self, dataset) is not None:
                                subset = split_data.pop(0)
                                if isinstance(subset, Dataset):
                                    subset = subset.dataset.iloc[subset.indices].copy()
                                setattr(self, dataset, subset)

                    else:
                        raise ValueError('`train_val_test_split` must be a sequence of numbers or None'
                                         '(float for percentages and int for sample numbers) '
                                         'if both `data_file` and `split` have been specified. '
                                         f'Got {self.train_val_test_split}')

                elif (all(isinstance(split, (str, list, ListConfig, dict, DictConfig, type(None))) for split in self.train_val_test_split)
                      and not any([self.data_file, self.split])):
                    for dataset in ['train_data', 'val_data', 'test_data']:
                        data = getattr(self, dataset)
                        if data is not None:
                            setattr(self, dataset, process_data_file(data, self.data_dir, self.extra_keys))

                else:
                    raise ValueError('For training, you must specify either all of `data_file`, `split`, '
                                     'and `train_val_test_split` as a sequence of numbers or '
                                     'solely `train_val_test_split` as a sequence of data file paths.')

            else:
                raise ValueError("For training, you must specify `train_val_test_split`. "
                                 "For testing/predicting, you must specify only `data_file` without "
                                 "`train_val_test_split` or `split`.")

        unique_smiles = []
        if isinstance(self.featurizers['drug'], BaseFeatureLoader):
            if not getattr(self.featurizers['drug'], 'feature_map'):
                for data_name in data_list:
                    data = getattr(self, data_name)
                    if isinstance(data, pd.DataFrame):
                        unique_smiles.extend(list(data['X1^'].unique()))
                    elif isinstance(data, list):
                        for df in data:
                            unique_smiles.extend(list(df['X1^'].unique()))
                    elif isinstance(data, dict):
                        for df in data.values():
                            unique_smiles.extend(list(df['X1^'].unique()))
                unique_smiles = set(unique_smiles)

        unique_fasta = []
        if isinstance(self.featurizers['protein'], BaseFeatureLoader):
            if not getattr(self.featurizers['protein'], 'feature_map'):
                for data_name in data_list:
                    data = getattr(self, data_name)
                    if isinstance(data, pd.DataFrame):
                        unique_fasta.extend(list(data['X2'].unique()))
                    elif isinstance(data, list):
                        for df in data:
                            unique_fasta.extend(list(df['X2'].unique()))
                    elif isinstance(data, dict):
                        for df in data.values():
                            unique_fasta.extend(list(df['X2'].unique()))
                unique_fasta = set(unique_fasta)

        self.featurizers = {
            'drug': ensure_cached(self.featurizers['drug'], unique_smiles),
            'protein': ensure_cached(self.featurizers['protein'], unique_fasta)
        }

        for data_name in data_list:
            data = getattr(self, data_name)
            if isinstance(data, pd.DataFrame):
                setattr(self, data_name,
                    self.dataset(
                        df=data,
                        drug_featurizer=self.featurizers['drug'],
                        protein_featurizer=self.featurizers['protein']
                    )
                )
            elif isinstance(data, list):
                for i, child_data in enumerate(data):
                    if isinstance(child_data, pd.DataFrame):
                        data[i] = self.dataset(
                            df=child_data,
                            drug_featurizer=self.featurizers['drug'],
                            protein_featurizer=self.featurizers['protein']
                        )
                setattr(self, data_name, data)
            elif isinstance(data, dict):
                for key, child_data in data.items():
                    if isinstance(child_data, pd.DataFrame):
                        data[key] = self.dataset(
                            df=child_data,
                            drug_featurizer=self.featurizers['drug'],
                            protein_featurizer=self.featurizers['protein']
                        )
                setattr(self, data_name, data)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_sampler=SafeBatchSampler(
                data_source=self.train_data,
                batch_size=self.batch_size,
                # Dropping the last batch prevents problems caused by variable batch sizes in training, e.g.,
                # batch_size=1 in BatchNorm, and shuffling ensures the model be trained on all samples over epochs.
                drop_last=True,
                shuffle=True,
            ),
            # batch_size=self.batch_size,
            # shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            persistent_workers=True if self.num_workers > 0 else False,
            # worker_init_fn = worker_init_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_sampler=SafeBatchSampler(
                data_source=self.val_data,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False
            ),
            # batch_size=self.batch_size,
            # shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            persistent_workers=True if self.num_workers > 0 else False,
            # worker_init_fn = worker_init_fn
        )

    def test_dataloader(self):
        test_loader = partial(DataLoader,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            persistent_workers=True if self.num_workers > 0 else False,
            # worker_init_fn = worker_init_fn
        )
        safe_batch_sampler = partial(SafeBatchSampler,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False
        )

        if isinstance(self.test_data, list):
            return [test_loader(dataset=dataset, batch_sampler=safe_batch_sampler(data_source=dataset)) for dataset in self.test_data]
        elif isinstance(self.test_data, dict):
            return {key: test_loader(dataset=dataset, batch_sampler=safe_batch_sampler(data_source=dataset)) for key, dataset in self.test_data.items()}
        else:
            return test_loader(dataset=self.test_data, batch_sampler=safe_batch_sampler(data_source=self.test_data))

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_data,
            batch_sampler=SafeBatchSampler(
                data_source=self.predict_data,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False
            ),
            # batch_size=self.batch_size,
            # shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            persistent_workers=True if self.num_workers > 0 else False,
            # worker_init_fn = worker_init_fn
        )

    def teardown(self, stage: Optional[str] = None):
        for key, featurizer in self.featurizers.items():
            cache_clear(featurizer)

    def state_dict(self):
        state = {'featurizers': {}}
        for key, featurizer in self.featurizers.items():
            if hasattr(featurizer, 'get_state'):
                featurizer_dict = featurizer.get_state()
                for k, v in featurizer_dict.items():
                    if isinstance(v, defaultdict):
                        featurizer_dict[k] = dict(v)
                state['featurizers'][key] = featurizer_dict

        return state

    def load_state_dict(self, state_dict):
        if state_dict.get('featurizers'):
            for key, featurizer in self.featurizers.items():
                if hasattr(featurizer, 'set_state'):
                    featurizer.set_state(state_dict['featurizers'][key])
