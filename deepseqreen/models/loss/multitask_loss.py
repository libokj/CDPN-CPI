from typing import Dict, Literal

import torch


class MultitaskLoss(torch.nn.Module):
    """A generic multitask loss class that takes a dict of loss functions as input"""
    def __init__(self, losses: Dict, reduction=Literal['sum', 'mean', 'none']):
        super().__init__()
        self.n_tasks = len(losses)  # The number of tasks is equal to the number of loss functions
        self.losses = losses
        self.reduction = reduction

    def forward(self, preds, target):
        if isinstance(preds, torch.Tensor):
            preds = {'label': preds}
        if isinstance(target, torch.Tensor):
            target = {'label': target}
        # compute the weighted losses for each task by applying the corresponding loss function and weight
        losses = []
        for key, loss_fn in self.losses.items():
            if key in preds and key in target:
                loss = loss_fn(preds[key], target[key].float())
            elif key in preds:
                loss = loss_fn(preds[key])
            else:
                loss = loss_fn(target[key].float())
            losses.append(loss)

        if self.reduction == 'sum':
            return sum(losses)
        elif self.reduction == 'mean':
            return sum(losses) / self.n_tasks
        elif self.reduction == 'none':
            return losses


class MultitaskWeightedLoss(MultitaskLoss):
    """A multitask loss class that takes a tuple of loss functions and weights as input"""

    def __init__(self, losses: Dict, weights: Dict, reduction=Literal['sum', 'none']):
        super().__init__(losses)
        self.weights = weights
        self.reduction = reduction

    def forward(self, preds, target):
        if isinstance(preds, torch.Tensor):
            preds = {'label': preds}
        if isinstance(target, torch.Tensor):
            target = {'label': target}
        # compute the weighted losses for each task by applying the corresponding loss function and weight
        losses = []
        for key, loss_fn in self.losses.items():
            if key in preds and key in target:
                loss = self.weights.get(key, 1) * loss_fn(preds[key], target[key].float())

            elif key in preds:
                loss = self.weights.get(key, 1) * loss_fn(preds[key])

            else:
                loss = self.weights.get(key, 1) * loss_fn(target[key].float())

            losses.append(loss)

        if self.reduction == 'sum':
            return sum(losses)
        elif self.reduction == 'none':
            return losses
