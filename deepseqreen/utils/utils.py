# import time
# from pathlib import Path
# from typing import Any, Dict, List
#
# import hydra
# from pytorch_lightning import Callback
# from pytorch_lightning.loggers import Logger
# from pytorch_lightning.utilities import rank_zero_only
import functools
import warnings
from importlib.util import find_spec
from typing import Callable

from omegaconf import DictConfig

from deepseqreen.utils import get_logger, enforce_tags, print_config_tree

log = get_logger(__name__)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before a job is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)


def job_wrapper(extra_utils: bool) -> Callable:
    """Optional decorator that controls the failure behavior and extra utilities when executing a job function.

    This wrapper can be used to:
    - make sure loggers are closed even if the job function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.job_wrapper(extra_utils)
    def train(cfg: DictConfig) -> Tuple[dict, dict]:

        .

        return metric_dict, object_dict
    ```
    """
    def decorator(job_func):
        def wrapped_func(cfg: DictConfig):
            # execute the job
            try:
                # apply extra utilities
                if extra_utils:
                    extras(cfg)
                metric_dict, object_dict = job_func(cfg=cfg)

            # things to do if exception occurs
            except Exception as ex:
                # save exception to `.log` file
                log.exception("")

                # some hyperparameter combinations might be invalid or cause out-of-memory errors
                # so when using hparam search plugins like Optuna, you might want to disable
                # raising the below exception to avoid multirun failure
                raise ex

            # things to always do after either success or exception
            finally:
                # display output dir path in terminal
                log.info(f"Output dir: {cfg.paths.output_dir}")

                # always close wandb run (even if exception occurs so multirun won't fail)
                if find_spec("wandb"):  # check if wandb is installed
                    import wandb

                    if wandb.run:
                        log.info("Closing wandb!")
                        wandb.finish()

            return metric_dict, object_dict
        return wrapped_func
    return decorator


# def stateful(**state_vars):
#     def decorator(func):
#         # Initialize the state variables
#         func._state_vars = {k: v() if callable(v) else v for k, v in state_vars.items()}
#
#         def wrapper(*args, **kwargs):
#             # Inject the state variables into the function's local namespace
#             for key, value in func._state_vars.items():
#                 setattr(wrapper, key, value)
#             return func(*args, **kwargs)
#
#         # Add methods to get and set the state
#         def get_state():
#             return func._state_vars
#
#         def set_state(state):
#             func._state_vars.update(state)
#             for key, value in state.items():
#                 setattr(wrapper, key, value)
#
#         wrapper.get_state = get_state
#         wrapper.set_state = set_state
#
#         return wrapper
#     return decorator
class Stateful(object):
    def get_state(self):
        state = {}
        for key, value in self.__dict__.items():
            if not key.startswith('__') and not callable(value):
                state[key] = value
        return state

    def set_state(self, state_dict):
        self.__dict__.update(state_dict)

# @rank_zero_only
# def save_file(path, content) -> None:
#     """Save file in rank zero mode (only on one process in multi-GPU setup)."""
#     with open(path, "w+") as file:
#         file.write(content)
#
#
# def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
#     """Instantiates callbacks from config."""
#     callbacks: List[Callback] = []
#
#     if not callbacks_cfg:
#         log.warning("Callbacks config is empty.")
#         return callbacks
#
#     if not isinstance(callbacks_cfg, DictConfig):
#         raise TypeError("Callbacks config must be a DictConfig!")
#
#     for _, cb_conf in callbacks_cfg.items():
#         if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
#             log.info(f"Instantiating callback <{cb_conf._target_}>")
#             callbacks.append(hydra.utils.instantiate(cb_conf))
#
#     return callbacks
#
#
# def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
#     """Instantiates loggers from config."""
#     logger: List[Logger] = []
#
#     if not logger_cfg:
#         log.warning("Logger config is empty.")
#         return logger
#
#     if not isinstance(logger_cfg, DictConfig):
#         raise TypeError("Logger config must be a DictConfig!")
#
#     for _, lg_conf in logger_cfg.items():
#         if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
#             log.info(f"Instantiating logger <{lg_conf._target_}>")
#             logger.append(hydra.utils.instantiate(lg_conf))
#
#     return logger
#
#
# @rank_zero_only
# def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
#     """Controls which config parts are saved by lightning loggers.
#
#     Additionally saves:
#     - Number of model parameters
#     """
#
#     hparams = {}
#
#     cfg = object_dict["cfg"]
#     model = object_dict["model"]
#     trainer = object_dict["trainer"]
#
#     if not trainer.logger:
#         log.warning("Logger not found! Skipping hyperparameter logging.")
#         return
#
#     hparams["model"] = cfg["model"]
#
#     # TODO Accommodation for LazyModule
#     # save number of model parameters
#     hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
#     hparams["model/params/trainable"] = sum(
#         p.numel() for p in model.parameters() if p.requires_grad
#     )
#     hparams["model/params/non_trainable"] = sum(
#         p.numel() for p in model.parameters() if not p.requires_grad
#     )
#
#     hparams["data"] = cfg["data"]
#     hparams["trainer"] = cfg["trainer"]
#
#     hparams["callbacks"] = cfg.get("callbacks")
#     hparams["extras"] = cfg.get("extras")
#
#     hparams["job_name"] = cfg.get("job_name")
#     hparams["tags"] = cfg.get("tags")
#     hparams["ckpt_path"] = cfg.get("ckpt_path")
#     hparams["seed"] = cfg.get("seed")
#
#     # send hparams to all loggers
#     trainer.logger.log_hyperparams(hparams)


# def close_loggers() -> None:
#     """Makes sure all loggers closed properly (prevents logging failure during multirun)."""
#
#     log.info("Closing loggers.")
#
#     if find_spec("wandb"):  # if wandb is installed
#         import wandb
#
#         if wandb.run:
#             log.info("Closing wandb!")
#             wandb.finish()
