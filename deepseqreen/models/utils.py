import torch


def get_mask(v):
    """
    Generate a padding mask for an input tensor of shape [batch_size, sequence_length, ...] according to
    the Transformer conventions (1 for padding and 0 for valid tokens).
    To use the alternative definition, negate the output mask (`~mask`) in the forward method.
    """
    if v.shape[1] == 1:
        mask = torch.zeros(v.shape[:2], device=v.device).bool()
    elif hasattr(v, 'mask'):
        # Usually provided by torch_geometric.utils.to_dense_batch but with the mask negated to match transformer conventions.
        mask = v.mask
    elif hasattr(v, 'lengths'):
        # Usually provided by DeepSEQreen collation with automatic padding.
        mask = torch.arange(v.size(1), device=v.device) >= v.lengths.unsqueeze(1)
    else:
        mask = torch.zeros(v.shape[:2], device=v.device).bool()  # Mask all as valid

    return mask
