from copy import deepcopy
import inspect
from typing import Dict, Sequence, TypeVar, Union

from torch.utils.data import get_worker_info

from deepseqreen.data.utils.collator import collate_fn
from deepseqreen.data.utils.label import label_transform
from deepseqreen.data.utils.sampler import SafeBatchSampler
from deepseqreen.utils import get_logger


log = get_logger(__name__)

T = TypeVar('T')
FlexibleIterable = Union[T, Sequence[T], Dict[str, T]]


def convert_none_like_str_to_none(lst):
    none_like_strs_lower = ['none', 'null', 'nan', 'na', 'nil']
    return [None if str(x).lower() in none_like_strs_lower else x for x in lst]

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset

    for attr_name in dir(dataset):
        if attr_name.endswith("featurizer"):
            featurizer = getattr(dataset, attr_name)
            if hasattr(featurizer, 'in_memory') and featurizer.in_memory is False:
                log.info(f"Worker {worker_id}: Opening feature file...")
                featurizer.open()
