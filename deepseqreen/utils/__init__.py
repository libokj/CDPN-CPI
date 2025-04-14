from deepseqreen.utils.logging import get_logger, log_hyperparameters
from deepseqreen.utils.instantiators import instantiate_callbacks, instantiate_loggers
from deepseqreen.utils.rich import enforce_tags, print_config_tree
from deepseqreen.utils.utils import extras, job_wrapper


def passthrough(x):
    return x
