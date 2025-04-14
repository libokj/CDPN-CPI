import torch
from torch import Tensor
from torchmetrics.retrieval.base import RetrievalMetric
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs

from deepseqreen.models.metrics.rie import calc_rie


class BEDROC(RetrievalMetric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
            self,
            alpha: float = 80.5,
    ):
        super().__init__()
        self.alpha = alpha

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        preds, target = _check_retrieval_functional_inputs(preds, target)

        n_total = target.size(0)
        n_actives = target.sum()

        if n_actives == 0:
            return torch.tensor(0.0, device=preds.device)
        elif n_actives == n_total:
            return torch.tensor(1.0, device=preds.device)

        r_a = n_actives / n_total
        exp_a = torch.exp(torch.tensor(self.alpha))

        idx = torch.argsort(preds, descending=True, stable=True)
        active_ranks = torch.take(target, idx).nonzero() + 1

        rie = calc_rie(n_total, active_ranks, r_a, exp_a)
        rie_min = (1 - exp_a ** r_a) / (r_a * (1 - exp_a))
        rie_max = (1 - exp_a ** (-r_a)) / (r_a * (1 - exp_a ** (-1)))

        return (rie - rie_min) / (rie_max - rie_min)

    def plot(self, val=None, ax=None):
        return self._plot(val, ax)
