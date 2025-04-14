import numpy as np
from pathlib import Path

from deepseqreen.utils import get_logger

log = get_logger(__name__)


class BaseFeatureLoader:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.feature_map = {}

    def __call__(self, key):
        if key in self.feature_map:
            return self.feature_map[key]
        else:
            log.info(f"Feature for key {key} not found in feature map.")
            return None


class HDF5FeatureLoader(BaseFeatureLoader):
    def __init__(self, file_path, in_memory=True):
        super().__init__(file_path)
        self.file_handle = None
        self.in_memory = in_memory
        self.feature_map = {}

    def encode_name(self, name):
        return name.replace("/", "_slash_")

    def decode_name(self, name):
        return name.replace("_slash_", "/")

    def load_feature_map(self, keys={}):
        try:
            import h5py
        except ImportError:
            log.error("h5py is not installed. Please install it to use HDF5FeatureLoader.")
            raise
        with h5py.File(self.file_path, 'r') as f:
            keys = [self.encode_name(key) for key in keys] if len(keys) else f.keys()
            log.info(f"Loading features from {self.file_path}")
            for key in keys:
                self.feature_map[self.decode_name(key)] = f[key][:]

    def __call__(self, sequence):
        if self.in_memory:
            if sequence in self.feature_map:
                return self.feature_map[sequence]
            else:
                log.info(f"Feature for sequence {sequence} not found in feature map.")
                return None
        else:
            if self.file_handle is None:
                self.open()
            encoded_sequence = self.encode_name(sequence)
            if encoded_sequence in self.file_handle:
                return self.file_handle[encoded_sequence][:]
            else:
                log.info(f"Feature for sequence {sequence} not found in feature map.")
                return None

    def open(self):
        try:
            import h5py
        except ImportError:
            log.error("h5py is not installed. Please install it to use HDF5FeatureLoader.")
            raise

        log.info(f"Opening feature file: {self.file_path}")
        self.file_handle = h5py.File(self.file_path, 'r')

    def close(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None


class NPZFeatureLoader(BaseFeatureLoader):
    def __init__(self, file, in_memory=True):
        super().__init__(file)
        self.in_memory = in_memory
        if in_memory:
            self.feature_map = dict(np.load(self.file, allow_pickle=True))
        else:
            self.feature_map = np.load(self.file, allow_pickle=True, mmap_mode='r')

    def __call__(self, sequence):
        if sequence in self.feature_map:
            return self.feature_map[sequence]
        else:
            log.info(f"Feature for sequence {sequence} not found in feature map.")
            return None
