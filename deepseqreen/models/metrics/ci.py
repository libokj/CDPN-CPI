import torch
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["ConcordanceIndex.plot"]


class ConcordanceIndex(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.5
    plot_upper_bound: float = 1.0

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("num_concordant", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        _check_same_shape(preds, target)

        g = preds.unsqueeze(-1) - preds
        g = (g == 0) * 0.5 + (g > 0)

        f = (target.unsqueeze(-1) - target) > 0
        f = torch.tril(f, diagonal=0)

        self.num_concordant += torch.sum(torch.mul(g, f)).long()
        self.num_valid += torch.sum(f).long()

    def compute(self):
        return torch.where(self.num_valid == 0, 0.0, self.num_concordant / self.num_valid)

    def plot(self, val=None, ax=None):
        return self._plot(val, ax)
