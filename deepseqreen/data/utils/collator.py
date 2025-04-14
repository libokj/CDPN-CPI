"""
Define collate functions for new data types here
"""
from functools import partial
from itertools import chain

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate_fn_map, collate_tensor_fn, collate


try:
    import torch_geometric
    def collate_pyg_fn(batch, collate_fn_map=None):
        return torch_geometric.data.Batch.from_data_list(batch)
    default_collate_fn_map[torch_geometric.data.data.BaseData] = collate_pyg_fn
except Exception:
    pass

try:
    import dgl
    def collate_dgl_fn(batch, collate_fn_map=None):
        return dgl.batch(batch)
    default_collate_fn_map[dgl.DGLGraph] = collate_dgl_fn
except Exception:
    pass


def pad_collate_tensor_fn(batch, padding_value=0.0, collate_fn_map=None):
    """
    Similar to pad_packed_sequence(pack_sequence(batch, enforce_sorted=False), batch_first=True),
    but additionally supports padding a list of square Tensors of size ``(L x L x ...)``.
    :param batch:
    :param padding_value:
    :param collate_fn_map:
    :return: padded_batch, lengths
    """
    batch = [tensor.unsqueeze(0) if tensor.dim() == 0 else tensor.squeeze() if tensor.dim() > 2 else tensor for tensor in batch]
    lengths = [tensor.size(0) for tensor in batch]
    if any(element != lengths[0] for element in lengths[1:]):
        try:
            # Tensors share at least one common dimension size, use pad_sequence
            batch = pad_sequence(batch, batch_first=True, padding_value=padding_value)
        except RuntimeError:
            # Tensors do not share any common dimension size, find the max size of each dimension in the batch
            max_sizes = [max([tensor.size(dim) for tensor in batch]) for dim in range(batch[0].dim())]
            # Pad every dimension of all tensors in the batch to be the respective max size with the value
            batch = collate_tensor_fn([
                torch.nn.functional.pad(
                    tensor, tuple(chain.from_iterable(
                        [(0, max_sizes[dim] - tensor.size(dim)) for dim in range(tensor.dim())][::-1])
                    ), mode='constant', value=padding_value) for tensor in batch
            ])
    else:
        batch = collate_tensor_fn(batch)

    lengths = torch.as_tensor(lengths)
    # Return the padded batch tensor and the lengths
    # return batch, lengths
    # batch.lengths = lengths
    return VariableLengthSequence(data=batch, lengths=lengths)


def collate_fn(batch, automatic_padding=False, padding_value=0):
    if automatic_padding:
        default_collate_fn_map.update({
            torch.Tensor: partial(pad_collate_tensor_fn, padding_value=padding_value),
        })
    return collate(batch, collate_fn_map=default_collate_fn_map)


# class VariableLengthSequence(torch.Tensor):
#     """
#     A custom PyTorch Tensor class that is similar to PackedSequence, except it can be directly used as a batch tensor,
#     and it has an attribute called lengths, which signifies the length of each original sequence in the batch.
#     """
#
#     def __new__(cls, data, lengths):
#         """
#         Creates a new VariableLengthSequence object from the given data and lengths.
#         Args:
#             data (torch.Tensor): The batch collated tensor of shape (batch_size, max_length, *).
#             lengths (torch.Tensor): The lengths of each original sequence in the batch of shape (batch_size,).
#         Returns:
#             VariableLengthSequence: A new VariableLengthSequence object.
#         """
#         # Check the validity of the inputs
#         assert isinstance(data, torch.Tensor), "data must be a torch.Tensor"
#         assert isinstance(lengths, torch.Tensor), "lengths must be a torch.Tensor"
#         assert data.dim() >= 2, "data must have at least two dimensions"
#         assert lengths.dim() == 1, "lengths must have one dimension"
#         assert data.size(0) == lengths.size(0), "data and lengths must have the same batch size"
#         assert lengths.min() > 0, "lengths must be positive"
#         assert lengths.max() <= data.size(1), "lengths must not exceed the max length of data"
#
#         # Create a new tensor object from data
#         obj = super().__new__(cls, data)
#
#         # Set the lengths attribute
#         obj.lengths = lengths
#
#         return obj


import torch

class VariableLengthSequence(torch.Tensor):
    def __new__(cls, data, lengths=None, *args, **kwargs):
        instance = torch.Tensor._make_subclass(cls, data, *args, **kwargs)
        instance._lengths = lengths if lengths is not None else torch.zeros(data.size(0), dtype=torch.long)
        return instance

    @property
    def lengths(self):
        return self._lengths

    @lengths.setter
    def lengths(self, lengths):
        if not isinstance(lengths, torch.Tensor):
            raise TypeError("lengths must be a torch.Tensor")
        self._lengths = lengths

    def clone(self, *args, **kwargs):
        lengths_clone = self.lengths.clone() if hasattr(self, '_lengths') else None
        return VariableLengthSequence(super().clone(*args, **kwargs), lengths_clone)

    def to(self, *args, **kwargs):
        return VariableLengthSequence(super().to(*args, **kwargs), self.lengths.to(*args, **kwargs))

    def cpu(self, *args, **kwargs):
        return VariableLengthSequence(super().cpu(*args, **kwargs), self.lengths.cpu(*args, **kwargs))

    def cuda(self, *args, **kwargs):
        return VariableLengthSequence(super().cuda(*args, **kwargs), self.lengths.cuda(*args, **kwargs))

    def pin_memory(self, *args, **kwargs):
        return VariableLengthSequence(super().pin_memory(*args, **kwargs), self.lengths.pin_memory(*args, **kwargs))

    def detach(self, *args, **kwargs):
        lengths_detach = self.lengths.detach(*args, **kwargs)
        return VariableLengthSequence(super().detach(*args, **kwargs), lengths_detach)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # Operations that should preserve lengths
        preserve_lengths_ops = {
            torch.Tensor.clone,
            torch.Tensor.to,
            torch.Tensor.cpu,
            torch.Tensor.cuda,
            torch.Tensor.pin_memory,
            torch.Tensor.detach,
        }

        # Dtype change operations
        dtype_ops = {
            torch.Tensor.float,
            torch.Tensor.double,
            torch.Tensor.half,
            torch.Tensor.long,
            torch.Tensor.int,
            torch.Tensor.short,
            torch.Tensor.char,
            torch.Tensor.byte,
            torch.Tensor.bool,
        }

        if func in preserve_lengths_ops or func in dtype_ops:
            result = super().__torch_function__(func, types, args, kwargs)
            if isinstance(result, torch.Tensor) and isinstance(args[0], VariableLengthSequence):
                return VariableLengthSequence(result, args[0].lengths)
            return result
        else:
            # Fall back to a standard tensor
            result = super().__torch_function__(func, types, args, kwargs)
            if isinstance(result, torch.Tensor):
                return result.as_subclass(torch.Tensor)
            return result
