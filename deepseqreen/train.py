from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import lightning
import hydra
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from deepseqreen.utils.hydra import checkpoint_rerun_config
from deepseqreen.utils import get_logger, job_wrapper, instantiate_callbacks, instantiate_loggers, log_hyperparameters
from deepseqreen.data.utils import convert_none_like_str_to_none


log = get_logger(__name__)


# def flatten_dict(d):
#     def _flatten(current_dict):
#         items = {}
#         for k, v in current_dict.items():
#             if isinstance(v, dict):
#                 flattened = _flatten(v)
#                 items.update(flattened)
#             elif isinstance(v, list):
#                 for i, item in enumerate(v):
#                     if isinstance(item, dict):
#                         flattened = _flatten(item)
#                         items.update(flattened)
#                     else:
#                         items[k] = item
#             else:
#                 items[k] = v
#         return items
#
#     return _flatten(d)


@job_wrapper(extra_utils=True)
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @job_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    # fix_dict_config(cfg)
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating callbacks.")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers.")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>.")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>.")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup('fit')

    log.info(f"Instantiating model <{cfg.model._target_}>.")
    if (ckpt_path := cfg.get("ckpt_path")) and cfg.get("finetune"):
        OmegaConf.set_struct(cfg, False)
        cfg.model._target_ = f"{cfg.model._target_}.load_from_checkpoint"
        cfg.model["checkpoint_path"] = ckpt_path
        ckpt_path = None

    # Retrieve the dynamic hyperparameters from datamodule and update the model config
    for featurizer_state in datamodule.state_dict()['featurizers'].values():
        for key, value in featurizer_state.items():
            if key in cfg.model.predictor:
                if isinstance(value, defaultdict):
                    value = dict(value)
                cfg['model']['predictor'][key] = value

    model: LightningModule = hydra.utils.instantiate(cfg.model)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    # Temporary fix to explicitly initialize UninitializedParameters in LazyModules
    # for batch in datamodule.train_dataloader():
    #     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #     batch = batch.to(device)
    #     model(batch)
    #     break

    if logger:
        log.info("Logging hyperparameters...")
        log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model...")
        model = torch.compile(model)

    log.info("Start training...")

    try:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    except Exception as e:
        log.exception("Training failed due to an error.", exc_info=e)
        raise

    if trainer.checkpoint_callback.best_model_path:
        ckpt_path = Path(trainer.checkpoint_callback.best_model_path).resolve()
        log.info(f"Best checkpoint path: {ckpt_path}")
    else:
        ckpt_path = None
        log.warning("Best checkpoint not saved.")

    if convert_none_like_str_to_none(cfg.data.train_val_test_split)[2] is not None:
        log.info("Start testing...")
        if ckpt_path is None:
            log.warning("Best checkpoint not found. Using current weights for testing.")
        try:
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        except Exception as e:
            log.exception("Testing failed due to an error.", exc_info=e)
            raise

    metric_dict = trainer.callback_metrics
    metric_dict['ckpt_path'] = ckpt_path

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    if cfg.get('ckpt_path'):
        cfg = checkpoint_rerun_config(cfg)
    metric_dict, _ = train(cfg)
    cfg.ckpt_path = metric_dict.get('ckpt_path')

    objective_metrics = cfg.get("objective_metrics")

    if not objective_metrics:
        return None

    else:
        invalid_metrics = [metric for metric in objective_metrics if metric not in metric_dict]

        if invalid_metrics:
            raise ValueError(
                f"Unable to find {invalid_metrics} (specified in `objective_metrics`) in `metric_dict`.\n"
                "Make sure your `model.metrics` and `sweep.objective_metrics` configs are correct."
            )

        # metric_value = metric_dict[objective_metric].item()
        metric_values = tuple([metric_dict[metric].item() for metric in objective_metrics])

        for objective_metric, metric_value in zip(objective_metrics, metric_values):
            log.info(f"Retrieved objective: {objective_metric}={metric_value}")

        # return optimized metrics
        return metric_values


if __name__ == "__main__":
    main()
