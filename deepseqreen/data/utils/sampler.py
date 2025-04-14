from typing import Mapping, Iterable

from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler


class SafeBatchSampler(BatchSampler):
    """
    A safe `batch_sampler` that skips samples with `None` values, supports shuffling, and keep a fixed batch size.

    Args:
        data_source (Dataset): The dataset to sample from.
        batch_size (int): The size of each batch.
        drop_last (bool): Whether to drop the last batch if its size is smaller than `batch_size`. Defaults to `False`.
        shuffle (bool, optional): Whether to shuffle the data before sampling. Defaults to `True`.

    Example:
        >>> dataloader = DataLoader(dataset, batch_sampler=SafeBatchSampler(dataset, batch_size, drop_last, shuffle))
    """
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool, sampler=None):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        if sampler:
            pass
        elif shuffle:
            sampler = RandomSampler(data_source)  # type: ignore[arg-type]
        else:
            sampler = SequentialSampler(data_source)  # type: ignore[arg-type]

        super().__init__(sampler, batch_size, drop_last)
        self.data_source = data_source

    # def __iter__(self):
    #     batch = []
    #     for idx in self.sampler:
    #         sample = self.data_source[idx]
    #         # if isinstance(sample, list | tuple):
    #         #     pass
    #         # elif isinstance(sample, dict):
    #         #     sample = sample.values()
    #         # elif isinstance(sample, Series):
    #         #     sample = sample.values
    #         # else:
    #         #     sample = [sample]
    #         if isinstance(sample, (Iterable, Mapping)) and not isinstance(sample, str):
    #             if isinstance(sample, Mapping):
    #                 sample = list(sample.values())
    #         else:
    #             sample = [sample]
    #
    #         if all(v is not None for v in sample):
    #             batch.append(idx)
    #             if len(batch) == self.batch_size:
    #                 yield batch
    #                 batch = []
    #
    #     if len(batch) > 0 and not self.drop_last:
    #         yield batch
    #
    #     if not batch:
    #         raise StopIteration

    def __iter__(self):
        batch = [0] * self.batch_size
        idx_in_batch = 0
        for idx in self.sampler:
            sample = self.data_source[idx]

            if isinstance(sample, Mapping):
                values = sample.values()
            elif isinstance(sample, Iterable) and not isinstance(sample, str):
                values = sample
            else:
                values = [sample]

            if all(v is not None for v in values):
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size

        if idx_in_batch > 0 and not self.drop_last:
            yield batch[:idx_in_batch]

#        if not any(batch):
            # raise StopIteration
#            return
    def __len__(self):
        float("inf")
