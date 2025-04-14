from collections import defaultdict
from typing import List, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
from lightning import LightningDataModule, LightningModule, Trainer, Callback
import torch

from deepseqreen.utils.hydra import checkpoint_rerun_config
from deepseqreen.utils import get_logger, job_wrapper, instantiate_callbacks, instantiate_loggers

log = get_logger(__name__)


@job_wrapper(extra_utils=True)
def predict(cfg: DictConfig) -> Tuple[list, dict]:
    """Predict given checkpoint on a data predict set.

    This method is wrapped in optional @job_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    log.info("Instantiating callbacks.")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers.")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    if (ckpt_path := cfg.get("ckpt_path")):
        cfg.data._target_ = f"{cfg.data._target_}.load_from_checkpoint"
        OmegaConf.update(cfg, 'data.checkpoint_path', ckpt_path, force_add=True)

    log.info(f"Instantiating data <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup('predict')

    # Retrieve the dynamic hyperparameters from datamodule and update the model config
    for featurizer_state in datamodule.state_dict()['featurizers'].values():
        for key, value in featurizer_state.items():
            if key in cfg.model.predictor:
                if isinstance(value, defaultdict):
                    value = dict(value)
                cfg['model']['predictor'][key] = value

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    if cfg.get("compile"):
        log.info("Compiling model...")
        model = torch.compile(model)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "trainer": trainer,
    }

    log.info("Start predicting.")
    predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path, return_predictions=True)

    return predictions, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig):
    assert cfg.ckpt_path, "Checkpoint path (`ckpt_path`) must be specified for predicting."
    cfg = checkpoint_rerun_config(cfg)
    predictions, _ = predict(cfg)
    return predictions


if __name__ == "__main__":
    main()
