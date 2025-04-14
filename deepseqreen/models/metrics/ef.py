import math

from torch import Tensor, topk
from torchmetrics.retrieval.base import RetrievalMetric
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs


class EnrichmentFactor(RetrievalMetric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
            self,
            alpha: float,
    ):
        super().__init__()
        if alpha <= 0 or alpha > 1:
            raise ValueError(f"Argument ``alpha`` has to be in interval (0, 1] but got {alpha}")
        self.alpha = alpha

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        preds, target = _check_retrieval_functional_inputs(preds, target)

        n_total = target.size(0)
        n_sampled = math.ceil(n_total * self.alpha)
        _, idx = topk(preds, n_sampled)
        hits_sampled = target[idx].sum()
        hits_total = target.sum()

        return hits_sampled / (hits_total * self.alpha)

    def plot(self, val=None, ax=None):
        return self._plot(val, ax)
