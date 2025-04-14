import torch
from torch import Tensor
from torchmetrics.retrieval.base import RetrievalMetric
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs


def calc_rie(n_total, active_ranks, r_a, exp_a):
    numerator = (exp_a ** (- active_ranks / n_total)).sum()
    denominator = (1 - exp_a ** (-1)) / (exp_a ** (1 / n_total) - 1)

    return numerator / (r_a * denominator)


class RIE(RetrievalMetric):
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

        r_a = n_actives / n_total
        exp_a = torch.exp(torch.tensor(-self.alpha))

        idx = torch.argsort(preds, descending=True, stable=True)
        active_ranks = torch.take(target, idx).nonzero() + 1

        return calc_rie(n_total, active_ranks, r_a, exp_a)

    def plot(self, val=None, ax=None):
        return self._plot(val, ax)
