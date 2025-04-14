"""
DeepScreen package initialization, registering custom objects and monkey patching for some libraries.
"""
from collections import defaultdict
import sys
from builtins import eval

import lightning.fabric.strategies.launchers.subprocess_script as subprocess_script
from lightning.pytorch.trainer.connectors.checkpoint_connector import _CheckpointConnector
from lightning.pytorch.trainer.states import TrainerFn
import torch
from omegaconf import OmegaConf

from deepseqreen.utils import get_logger

log = get_logger(__name__)

# Allow basic Python operations in hydra interpolation; examples:
# `in_channels: ${eval:${model.drug_encoder.out_channels}+${model.protein_encoder.out_channels}}`
# `subdir: ${eval:${hydra.job.override_dirname}.replace('/', '.')}`
OmegaConf.register_new_resolver("eval", eval)


def sanitize_path(path_str: str):
    """
    Sanitize a string for path creation by replacing unsafe characters and cutting length to 255 (OS limitation).
    """
    return path_str.replace("/", ".").replace("\\", ".").replace(":", "-")[:255]


OmegaConf.register_new_resolver("sanitize_path", sanitize_path)


def _hydra_subprocess_cmd(local_rank: int):
    """
    Monkey patching for lightning.fabric.strategies.launchers.subprocess_script._hydra_subprocess_cmd
    Temporarily fixes the problem of unnecessarily creating log folders for DDP subprocesses in Hydra multirun/sweep.
    """
    import __main__  # local import to avoid https://github.com/Lightning-AI/lightning/issues/15218
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd, to_absolute_path

    # when user is using hydra find the absolute path
    if __main__.__spec__ is None:  # pragma: no-cover
        command = [sys.executable, to_absolute_path(sys.argv[0])]
    else:
        command = [sys.executable, "-m", __main__.__spec__.name]

    command += sys.argv[1:]

    cwd = get_original_cwd()
    rundir = f'"{HydraConfig.get().runtime.output_dir}"'
    # Set output_subdir null since we don't want different subprocesses trying to write to config.yaml
    command += [f"hydra.job.name=train_ddp_process_{local_rank}",
                "hydra.output_subdir=null,"
                f"hydra.runtime.output_dir={rundir}"]
    return command, cwd


subprocess_script._hydra_subprocess_cmd = _hydra_subprocess_cmd

# from torch import Tensor
# from lightning.fabric.utilities.distributed import _distributed_available
# from lightning.pytorch.utilities.rank_zero import WarningCache
# from lightning.pytorch.utilities.warnings import PossibleUserWarning
# from lightning.pytorch.trainer.connectors.logger_connector.result import _ResultCollection

# warning_cache = WarningCache()
#
# @staticmethod
# def _get_cache(result_metric, on_step: bool):
#     cache = None
#     if on_step and result_metric.meta.on_step:
#         cache = result_metric._forward_cache
#     elif not on_step and result_metric.meta.on_epoch:
#         if result_metric._computed is None:
#             should = result_metric.meta.sync.should
#             if not should and _distributed_available() and result_metric.is_tensor:
#                 warning_cache.warn(
#                     f"It is recommended to use `self.log({result_metric.meta.name!r}, ..., sync_dist=True)`"
#                     " when logging on epoch level in distributed setting to accumulate the metric across"
#                     " devices.",
#                     category=PossibleUserWarning,
#                 )
#             result_metric.compute()
#             result_metric.meta.sync.should = should
#
#         cache = result_metric._computed
#
#         if cache is not None:
#             if isinstance(cache, Tensor):
#                 if not result_metric.meta.enable_graph:
#                     return cache.detach()
#
#     return cache
#
#
# _ResultCollection._get_cache = _get_cache

if torch.cuda.is_available():
    if torch.cuda.get_device_capability() >= (8, 0):
        torch.set_float32_matmul_precision("high")
        log.info("Your GPU supports tensor cores, "
                 "we will enable it automatically by setting `torch.set_float32_matmul_precision('high')`")

def _restore_modules_and_callbacks(self, checkpoint_path=None) -> None:
    # restore modules after setup
    self.resume_start(checkpoint_path)
    self.restore_datamodule()
    self.restore_model()
    if self.trainer.state.fn == TrainerFn.FITTING:
        # restore callback states
        self.restore_callbacks()

_CheckpointConnector._restore_modules_and_callbacks = _restore_modules_and_callbacks
