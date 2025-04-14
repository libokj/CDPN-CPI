from functools import partial
from typing import Optional, Sequence, Dict

import torch
from torch import nn, optim, Tensor
from lightning import LightningModule
from torchmetrics import Metric, MetricCollection

from deepseqreen.models.loss.multitask_loss import MultitaskLoss
from deepseqreen.utils import get_logger


log = get_logger(__name__)


class DTILightningModule(LightningModule):
    """
        Drug Target Interaction Prediction

        optimizer: a partially or fully initialized instance of class torch.optim.Optimizer
        drug_encoder: a fully initialized instance of class torch.nn.Module
        protein_encoder: a fully initialized instance of class torch.nn.Module
        classifier: a fully initialized instance of class torch.nn.Module
        model: a fully initialized instance of class torch.nn.Module
        metrics: a list of fully initialized instances of class torchmetrics.Metric
    """
    extra_return_keys = ['ID1', 'X1', 'ID2', 'X2', 'N', 'Y']

    def __init__(
            self,
            predictor: nn.Module,
            metrics: Optional[Dict[str, Metric]] = (),
            optimizer: optim.Optimizer = None,
            lr_scheduler: Optional[Dict] = None,
            out: nn.Module = None,
            loss: nn.Module = None,
            activation: nn.Module = None,
    ):
        super().__init__()

        self.predictor = predictor
        self.out = out
        self.loss = loss
        self.activation = activation

        self.metrics = MetricCollection(dict(metrics))
        self.train_metrics = self.metrics.clone(prefix="train/")
        self.val_metrics = self.metrics.clone(prefix="val/")
        self.test_metrics = nn.ModuleList([])

        self.save_hyperparameters(logger=False)

    def setup(self, stage):
        if stage == 'fit':
            if any(isinstance(p, nn.parameter.UninitializedParameter) for p in self.parameters()):
                dataloader = self.trainer.datamodule.train_dataloader()
                dummy_batch = next(iter(dataloader))
                with torch.no_grad():
                    self.forward(dummy_batch)

        if stage ==  'test':
            dataloader = self.trainer.datamodule.test_dataloader()
            if isinstance(dataloader, dict):
                for key in dataloader.keys():
                    self.test_metrics.append(self.metrics.clone(prefix=f'test/{key}/'))
            elif isinstance(dataloader, list):
                for i in range(len(dataloader)):
                    self.test_metrics.append(self.metrics.clone(prefix=f'test/{i}/'))
            else:
                self.test_metrics.append(self.metrics.clone(prefix='test/'))

    def forward(self, batch):
        output = self.predictor(batch['X1^'], batch['X2^'])
        target = {'label': batch.get('Y')}
        indexes = batch.get('ID^')
        # weights = batch.get('weight')
        loss = None

        if isinstance(output, Tensor):
            output = {'label': output}

        extras = {k: output[k] for k in output if k not in ['label']}
        output['label'] = self.out(output['label']).squeeze(1)
        preds = self.activation(output['label'])
        if target['label'] is not None:
            if isinstance(self.loss, MultitaskLoss):
                target = target | {
                    key: batch[key] for key in self.loss.losses.keys() if key in batch
                }
                loss = self.loss(output, target)
            else:
                loss = self.loss(output['label'], target['label'].float())
        # if target is not None:
            # if weights is None:
            #    loss = self.loss(output, target.float())
            # else:
            #    loss = self.loss(output, target.float(), weights)
            # loss = self.loss(output, target.float())

        return preds, target['label'], indexes, loss, extras

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        preds, target, indexes, loss, _ = self.forward(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_metrics.update(preds=preds, target=target, indexes=indexes.long())

        return_dict = {
            'Y^': preds,
            'Y': target,
            'loss': loss
        }

        for key in self.extra_return_keys:
            if key in batch:
                return_dict[key] = batch[key]

        return return_dict

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        preds, target, indexes, loss, _ = self.forward(batch)

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_metrics.update(preds=preds, target=target, indexes=indexes.long())

        return_dict = {
            'Y^': preds,
            'Y': target,
            'loss': loss
        }

        for key in self.extra_return_keys:
            if key in batch:
                return_dict[key] = batch[key]

        return return_dict

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        preds, target, indexes, loss, extras = self.forward(batch)

        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=True)
        self.test_metrics[dataloader_idx].update(preds=preds, target=target, indexes=indexes.long())

        return_dict = {
            'Y^': preds,
            'Y': target,
            'loss': loss
        }

        for key in self.extra_return_keys:
            if key in batch:
                return_dict[key] = batch[key]

        return return_dict

    def on_test_epoch_end(self):
        for test_metric in self.test_metrics:
            self.log_dict(test_metric.compute())
            test_metric.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        preds, _, _, _, extras = self.forward(batch)
        for logger in self.loggers:
            if hasattr(logger.experiment, 'add_tensor'):
                for i in range(len(preds)):
                    for key, value in extras.items():
                        logger.experiment.add_tensor(
                            f"predict/{key}/{batch['X2'][i]}/{batch['X1'][i].replace('/', '_')}",
                            value[i]
                        )
        # return a dictionary for callbacks like BasePredictionWriter
        return_dict = {
            'Y^': preds,
        }

        for key in self.extra_return_keys:
            if key in batch:
                return_dict[key] = batch[key]

        return return_dict

    def configure_optimizers(self):
        optimizers_config = {'optimizer': self.hparams.optimizer(params=self.parameters())}
        if self.hparams.get('lr_scheduler'):
            optimizers_config['lr_scheduler'] = {
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            } | dict(self.hparams.lr_scheduler)
            optimizers_config['lr_scheduler']['scheduler'] = self.hparams.lr_scheduler.scheduler(
                optimizer=optimizers_config['optimizer']
            )

        return optimizers_config
