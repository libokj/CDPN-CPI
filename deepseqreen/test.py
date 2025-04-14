from collections import defaultdict
from typing import List, Tuple

import hydra
from lightning import LightningDataModule, LightningModule, Trainer, Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from deepseqreen.utils.hydra import checkpoint_rerun_config
from deepseqreen.utils import get_logger, job_wrapper, instantiate_callbacks, instantiate_loggers, log_hyperparameters


log = get_logger(__name__)


@job_wrapper(extra_utils=True)
def test(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule test set.

    This method is wrapped in optional @job_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    # fix_dict_config(cfg)

    log.info("Instantiating callbacks.")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers.")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>.")

    if (ckpt_path := cfg.get("ckpt_path")):
        cfg.data._target_ = f"{cfg.data._target_}.load_from_checkpoint"
        OmegaConf.update(cfg, 'data.checkpoint_path', ckpt_path, force_add=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup('test')

    # Retrieve the dynamic hyperparameters from datamodule and update the model config
    for featurizer_state in datamodule.state_dict()['featurizers'].values():
        for key, value in featurizer_state.items():
            if key in cfg.model.predictor:
                if isinstance(value, defaultdict):
                    value = dict(value)
                cfg['model']['predictor'][key] = value

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters.")
        log_hyperparameters(object_dict)

    log.info("Start testing.")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="test.yaml")
def main(cfg: DictConfig):
    assert cfg.ckpt_path, "Checkpoint path (`ckpt_path`) must be specified for testing."
    cfg = checkpoint_rerun_config(cfg)

    # evaluate the model
    metric_dict, _ = test(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    objective_metric = cfg.get("objective_metric")

    if not objective_metric:
        return None

    if objective_metric not in metric_dict:
        raise Exception(
            f"Unable to find `objective_metric` ({objective_metric}) in `metric_dict`.\n"
            "Make sure `objective_metric` name in `sweep` config is correct."
        )

    metric_value = metric_dict[objective_metric].item()
    log.info(f"Retrieved objective_metric. <{objective_metric}={metric_value}>")

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
