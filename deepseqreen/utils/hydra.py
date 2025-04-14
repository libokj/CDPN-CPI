from datetime import datetime
from pathlib import Path
import re
from typing import Any, Tuple

import pandas as pd
from filelock import SoftFileLock
from hydra import TaskFunction, compose, initialize_config_dir
from hydra.core.utils import _save_config
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.experimental.callbacks import Callback
from hydra.types import RunMode
from hydra._internal.defaults_list import Overrides
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.errors import MissingMandatoryValue

from deepseqreen.utils import get_logger

log = get_logger(__name__)


class CSVExperimentSummary(Callback):
    """
    On multirun end, aggregate the results from each job's metrics.csv and save them in metrics_summary.csv.
    """
    def __init__(self, filename: str = 'experiment_summary.csv', prefix: str | Tuple[str] = 'test/'):
        self.filename = filename
        self.prefix = prefix if isinstance(prefix, str) else tuple(prefix)
        self.input_experiment_summary = None
        self.time = {}
        self.lock = None
        self.run_mode = None
        self.summary_file_path = None

    def parse_ckpt_path(self, ckpt_path):
        parsed_ckpt_path = str(ckpt_path).strip("'\"")
        ckpt_list = []

        if parsed_ckpt_path.endswith(('.csv', '.txt', '.tsv', '.ssv', '.psv')):
            try:
                self.input_experiment_summary = pd.read_csv(parsed_ckpt_path, usecols=['ckpt_path'])
                self.input_experiment_summary['ckpt_path'] = self.input_experiment_summary['ckpt_path'].apply(
                    lambda x: x.strip("'\"") if type(x) is str else None
                )
                self.input_experiment_summary = self.input_experiment_summary[~self.input_experiment_summary['ckpt_path'].isna()]
                ckpt_list = list(set(self.input_experiment_summary['ckpt_path']))
                log.info(f"Using {len(ckpt_list)} checkpoint(s) from {parsed_ckpt_path}: {ckpt_list}.")
            except Exception as e:
                log.exception(
                    f'Error in parsing checkpoint paths from experiment summary ({parsed_ckpt_path}).',
                    exc_info=e
                )

        elif Path(parsed_ckpt_path).is_dir():
            try:
                ckpt_list = [str(file.resolve()) for file in Path(parsed_ckpt_path).glob('*.ckpt')]
                log.info(f"Using {len(ckpt_list)} checkpoint(s) from {parsed_ckpt_path}: {ckpt_list}.")
            except Exception as e:
                log.exception(
                    f'Error in parsing checkpoint paths from directory ({parsed_ckpt_path}).',
                    exc_info=e
                )

        if ckpt_list:
            parsed_ckpt_path = ','.join([f"'{ckpt}'" for ckpt in ckpt_list])
            return parsed_ckpt_path
        else:
            return ckpt_path

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        self.run_mode = RunMode.MULTIRUN
        self.summary_file_path = Path(config.hydra.sweep.dir) / self.filename

        if config.hydra.get('overrides') and config.hydra.overrides.get('task'):
            for i, override in enumerate(config.hydra.overrides.task):
                if override.startswith("ckpt_path"):
                    ckpt_path = override.split('=', 1)[1]
                    parsed_ckpt_path = self.parse_ckpt_path(ckpt_path)
                    config.hydra.overrides.task[i] = 'ckpt_path=' + parsed_ckpt_path
                    break
            if config.hydra.sweeper.get('params'):
                if config.hydra.sweeper.params.get('ckpt_path'):
                    config.hydra.sweeper.params.ckpt_path = self.parse_ckpt_path(config.hydra.sweeper.params.ckpt_path)

    def on_job_start(self, config: DictConfig, *, task_function: TaskFunction, **kwargs: Any) -> None:
        self.time['start'] = datetime.now()
        if config.hydra.mode == RunMode.RUN:
            self.run_mode = RunMode.RUN
            self.summary_file_path = Path(config.hydra.run.dir) / self.filename

    def on_job_end(self, config: DictConfig, job_return, **kwargs: Any) -> None:
        # Skip callback if job is DDP subprocess
        if "ddp" in job_return.hydra_cfg.hydra.job.name:
            return

        try:
            self.time['end'] = datetime.now()

            # Add job and override info
            info_dict = {}
            if job_return.overrides:
                info_dict = dict(override.split('=', 1) for override in job_return.overrides)
            info_dict['job_status'] = job_return.status.name
            if job_return.hydra_cfg.hydra.job.get('id'):
                info_dict['job_id'] = job_return.hydra_cfg.hydra.job.id
            info_dict['wall_time'] = str(self.time['end'] - self.time['start'])

            # Add checkpoint info
            if info_dict.get('ckpt_path'):
                info_dict['ckpt_path'] = str(info_dict['ckpt_path']).strip("'\"")

            ckpt_path = str(job_return.cfg.ckpt_path).strip("'\"")
            if Path(ckpt_path).is_file():
                if info_dict.get('ckpt_path') and ckpt_path != info_dict['ckpt_path']:
                    info_dict['previous_ckpt_path'] = info_dict['ckpt_path']
                info_dict['ckpt_path'] = ckpt_path
            if info_dict.get('ckpt_path'):
                if epoch_regex := re.search(r'epoch_(\d+)', info_dict['ckpt_path']):
                    info_dict['best_epoch'] = int(epoch_regex.group(1))

            # Add metrics info
            metrics_df = pd.DataFrame()
            if config.get('logger'):
                output_dir = Path(config.hydra.runtime.output_dir).resolve()
                csv_metrics_path = output_dir / config.logger.csv.name / "metrics.csv"
                if csv_metrics_path.is_file():
                    log.info(f"Summarizing metrics with prefix `{self.prefix}` from {csv_metrics_path}")
                    metrics_df = pd.read_csv(csv_metrics_path)
                    # Find rows where 'test/' columns are not null and reset its epoch to the best model epoch
                    if info_dict.get('best_epoch'):
                        if not (best_columns := [col for col in metrics_df.columns if col.startswith('test/')]):
                            if not (best_columns := [col for col in metrics_df.columns if col.startswith('val/')]):
                                best_columns = [col for col in metrics_df.columns if col.startswith('train/')]
                        mask = metrics_df[best_columns].notna().any(axis=1)
                        metrics_df.loc[mask, 'epoch'] = info_dict['best_epoch']
                        # Group and filter by best epoch
                        metrics_df = metrics_df.groupby('epoch').first().reset_index()
                        metrics_df = metrics_df[metrics_df['epoch'] == info_dict['best_epoch']]
                    else:
                        metrics_df = metrics_df.groupby('epoch').first().reset_index()
                        metrics_df = metrics_df[metrics_df['epoch'] == metrics_df['epoch'].max()]
                else:
                    log.info(f"No metrics.csv found in {output_dir}")

            if metrics_df.empty:
                metrics_df = pd.DataFrame(data=info_dict, index=[0])
            else:
                metrics_df = metrics_df.assign(**info_dict)
                metrics_df.index = [0]

            # Add extra info from the input batch experiment summary
            if self.input_experiment_summary is not None and 'ckpt_path' in metrics_df.columns:
                orig_meta = self.input_experiment_summary[
                    self.input_experiment_summary['ckpt_path'] == metrics_df['ckpt_path'][0]
                    ].head(1)
                if not orig_meta.empty:
                    orig_meta.index = [0]
                metrics_df = metrics_df.astype('O').combine_first(orig_meta.astype('O'))

            # Save the experiment summary
            if self.run_mode == RunMode.MULTIRUN:
                if not self.lock:
                    self.lock = SoftFileLock(str(self.summary_file_path) + ".lock")

                with self.lock:
                    if Path(self.summary_file_path).is_file():
                        summary_df = pd.concat([pd.read_csv(self.summary_file_path), metrics_df])
                    else:
                        summary_df = metrics_df

                    summary_df.dropna(inplace=True, axis=1, how='all')  # Drop empty columns
                    summary_df.to_csv(self.summary_file_path, index=False, mode='w')

            if self.run_mode == RunMode.RUN:
                metrics_df.dropna(inplace=True, axis=1, how='all')  # Drop empty columns
                metrics_df.to_csv(self.summary_file_path, index=False, mode='w')

            log.info(f"Experiment summary saved to {self.summary_file_path}")

        except Exception as e:
            log.exception("Unable to save the experiment summary due to an error.", exc_info=e)


def checkpoint_rerun_config(config: DictConfig):
    hydra_cfg = HydraConfig.get()
    hydra_instance = GlobalHydra.instance()
    ckpt_ignore_keys = [
        'data.data_file', 'data.split', 'data.train_val_test_split',
        'model.scheduler', 'callbacks',
        'trainer.devices', 'trainer.accelerator', 'trainer.precision',
        'trainer.limit_val_batches', 'trainer.num_sanity_val_steps'
    ]
    preset_specified = any([override.startswith('preset=') for override in hydra_cfg.overrides.task])
    if hydra_cfg.get('output_subdir') and not preset_specified:
        ckpt_cfg_path = Path(config.ckpt_path).parents[1] / hydra_cfg.output_subdir / 'config.yaml'
        hydra_output = Path(hydra_cfg.runtime.output_dir) / hydra_cfg.output_subdir

        if ckpt_cfg_path.is_file():
            log.info(f"Found config file for the checkpoint at {str(ckpt_cfg_path)}; "
                     f"merging config overrides with checkpoint config...")
            ckpt_cfg = OmegaConf.load(ckpt_cfg_path)

            for key in ckpt_ignore_keys:
                key_parts = key.split('.')
                sub_config = ckpt_cfg

                for part in key_parts[:-1]:
                    if not isinstance(sub_config, DictConfig):
                        break
                    sub_config = sub_config.get(part, None)

                if isinstance(sub_config, DictConfig):
                    if key_parts[-1] in sub_config:
                        del sub_config[key_parts[-1]]

            # Recompose checkpoint config with overrides
            if hydra_cfg.overrides.get('task'):
                config_loader = hydra_instance.config_loader()
                parsed_overrides, caching_repo = config_loader._parse_overrides_and_create_caching_repo(
                    hydra_cfg.job.config_name, hydra_cfg.overrides.task
                )
                overrides = Overrides(repo=caching_repo, overrides_list=parsed_overrides)
                config_loader._apply_overrides_to_config(overrides.config_overrides, ckpt_cfg)

                for key in ckpt_cfg.keys():
                    OmegaConf.update(config, key, ckpt_cfg[key], merge=True, force_add=True)
                _save_config(config, "config.yaml", hydra_output)

    return config
