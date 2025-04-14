from numbers import Number
from typing import Literal, Union, Sequence

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import Dataset

from deepseqreen.data.utils import label_transform, FlexibleIterable


class BaseEntityDataset(Dataset):
    def __init__(
            self,
            dataset_path: str,
            use_col_prefixes=('X', 'Y', 'ID', 'U')
    ):

        # Read the data table header row first to filter columns and create column dtype dict
        df = pd.read_csv(
            dataset_path,
            header=0, nrows=0,
            usecols=lambda col: col.startswith(use_col_prefixes)
        )
        # Read the whole data table
        df = pd.read_csv(
            dataset_path,
            header=0,
            usecols=df.columns,
            dtype={col: 'float32' if col.startswith('Y') else 'string' for col in df.columns}
        )

        self.df = df
        self.label_cols = [col for col in df.columns if col.startswith('Y')]
        self.label_unit_cols = [col for col in df.columns if col.startswith('U')]
        self.entity_id_cols = [col for col in df.columns if col.startswith('ID')]
        self.entity_cols = [col for col in df.columns if col.startswith('X')]

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        raise NotImplementedError


# TODO test transform
class SingleEntitySingleTargetDataset(BaseEntityDataset):
    def __init__(
            self,
            dataset_path: str,
            task: Literal['regression', 'binary', 'multiclass'],
            n_classes: int,
            featurizer: callable,
            transformer: TransformerMixin = None,
            thresholds: Union[Number, Sequence[Number]] = None,
            discard_intermediate: bool = None,
            forward_fill: bool = True
    ):
        super().__init__(dataset_path)

        assert len(self.entity_cols) == 1, 'The dataset contains more than 1 entity column (starting with `X`).'
        if len(self.label_cols) >= 0:
            assert len(self.label_cols) == 1, 'The dataset contains more than 1 label column (starting with `Y`).'
        # Remove trailing `1`s in column names for flexibility
        self.df.columns = self.df.columns.str.rstrip('1')

        # Forward-fill non-label columns
        nonlabel_cols = self.label_unit_cols + self.entity_id_cols + self.entity_cols
        if forward_fill:
            self.df[nonlabel_cols] = self.df[nonlabel_cols].ffill(axis=0)

        # Process target labels for training/testing if exist
        if self.label_cols:
            # Transform target labels
            self.df[self.label_cols] = self.df[self.label_cols].apply(
                label_transform,
                units=self.df.get('U', None),
                thresholds=thresholds,
                discard_intermediate=discard_intermediate).astype('float32')

            # Filter out rows with a NaN in Y (missing values); use inplace to save memory
            self.df.dropna(subset=self.label_cols, inplace=True)

            # Validate target labels
            # TODO: check sklearn.utils.multiclass.check_classification_targets
            match task:
                case 'regression':
                    assert all(self.df['Y'].apply(lambda x: isinstance(x, Number))), \
                        f"Y for task `regression` must be numeric; got {set(self.df['Y'].apply(type))}."
                case 'binary':
                    assert all(self.df['Y'].isin([0, 1])), \
                        f"Y for task `binary` (classification) must be 0 or 1, but Y got {pd.unique(self.df['Y'])}." \
                        "\nYou may set `thresholds` to discretize continuous labels."
                case 'multiclass':
                    assert n_classes >= 3, f'n_classes for task `multiclass` (classification) must be at least 3.'
                    assert all(self.df['Y'].apply(lambda x: x.is_integer() and x >= 0)), \
                        f"``Y` for task `multiclass` (classification) must be non-negative integers, " \
                        f"but `Y` got {pd.unique(self.df['Y'])}." \
                        "\nYou may set `thresholds` to discretize continuous labels."
                    target_n_unique = self.df['Y'].nunique()
                    assert target_n_unique == n_classes, \
                        f"You have set n_classes for task `multiclass` (classification) task to {n_classes}, " \
                        f"but `Y` has {target_n_unique} unique labels."

        if transformer:
            self.df['X'] = self.df['X'].apply(featurizer)
            try:
                check_is_fitted(transformer)
                self.df['X'] = list(transformer.transform(self.df['X']))
            except NotFittedError:
                self.df['X'] = list(transformer.fit_transform(self.df['X']))

            # Skip sample-wise feature extraction because it has already been done dataset-wise
            self.featurizer = lambda x: x

        self.featurizer = featurizer
        self.n_classes = n_classes
        self.df['ID'] = self.df.get('ID', self.df['X'])

    def __getitem__(self, idx):
        sample = self.df.loc[idx]
        return {
            'X': self.featurizer(sample['X']),
            'ID': sample['ID'],
            'Y': sample.get('Y')
        }


# TODO WIP
class MultiEntityMultiTargetDataset(BaseEntityDataset):
    def __init__(
            self,
            dataset_path: str,
            task: FlexibleIterable[Literal['regression', 'binary', 'multiclass']],
            n_class: FlexibleIterable[int],
            featurizers: FlexibleIterable[callable],
            thresholds: FlexibleIterable[Union[Number, Sequence[Number]]] = None,
            discard_intermediate: FlexibleIterable[bool] = None,
    ):
        super().__init__(dataset_path)
        label_col_prefix = tuple('Y')
        nonlabel_col_prefixes = tuple(('X', 'ID', 'U'))
        allowed_col_prefixes = label_col_prefix + nonlabel_col_prefixes

        # Read the headers first to filter columns and create column dtype dict
        df = pd.read_csv(
            dataset_path,
            header=0, nrows=0,
            usecols=lambda col: col.startswith(allowed_col_prefixes)
        )

        # Read the whole table
        df = pd.read_csv(
            dataset_path,
            header=0,
            usecols=df.columns,
            dtype={col: 'float32' if col.startswith('Y') else 'string' for col in df.columns}
        )
        label_cols = [col for col in df.columns if col.startswith(label_col_prefix)]
        nonlabel_cols = [col for col in df.columns if col.startswith(nonlabel_col_prefixes)]
        self.entity_cols = [col for col in nonlabel_cols if col.startswith('X')]

        # Forward-fill all non-label columns
        df[nonlabel_cols] = df[nonlabel_cols].ffill(axis=0)

        # Process target labels for training/testing
        if label_cols:
            # Transform target labels
            df[label_cols] = df[label_cols].apply(label_transform, units=df.get('U', None), thresholds=thresholds,
                                                  discard_intermediate=discard_intermediate).astype('float32')

            # Filter out rows with a NaN in Y (missing values)
            df.dropna(subset=label_cols, inplace=True)

            # Validate target labels
            # TODO: check sklearn.utils.multiclass.check_classification_targets
            # WIP
            match task:
                case 'regression':
                    assert all(df['Y'].apply(lambda x: isinstance(x, Number))), \
                        f"Y for task `regression` must be numeric; got {set(df['Y'].apply(type))}."
                case 'binary':
                    assert all(df['Y'].isin([0, 1])), \
                        f"Y for task `binary` must be 0 or 1, but Y got {pd.unique(df['Y'])}." \
                        "\nYou may set `thresholds` to discretize continuous labels."
                case 'multiclass':
                    assert len(label_cols) == len(n_class), \
                        (f'Data table has {len(label_cols)} label columns (`Y*`) but you have specified '
                         f'n_class of length {len(n_class)} for task `multiclass`.')
                    for label, n in zip(df[label_cols], n_class):
                        assert n >= 3, f'n_class for task `multiclass` must be at least 3.'
                        assert all(label.apply(lambda x: x.is_integer() and x >= 0)), \
                            f"Y for task `multiclass` must be non-negative integers, " \
                            f"but Y got {pd.unique(label)}." \
                            "\nYou may set `thresholds` to discretize continuous labels."
                        target_n_unique = label.nunique()
                        assert target_n_unique == n, \
                            f"You have set n_classes for task `multiclass` task to {n}, " \
                            f"but Y has {target_n_unique} unique labels."

        self.df = df
        self.featurizers = featurizers
        self.n_class = n_class

    def __len__(self):
        return len(self.df.index)

    # WIP
    def __getitem__(self, idx):
        sample = self.df.loc[idx]
        return {
            'X': [featurizer(x) for featurizer, x in zip(self.featurizers, sample[self.entity_cols])],
            'ID': sample.get('ID', sample['X']),
            'Y': sample.get('Y')
        }
