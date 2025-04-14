from pathlib import Path
from typing import Literal

from lightning.pytorch.callbacks import BasePredictionWriter
import pandas as pd
import torch

from deepseqreen.utils import get_logger

log = get_logger(__name__)


class CSVPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval: Literal["batch", "epoch"] = "batch"):
        super().__init__(write_interval)
        self.output_file = Path(output_dir, "predictions.csv")

    def setup(self, trainer, pl_module, stage: str):
        log.info(f"Saving predictions every {self.interval.value} for job `{stage}`.")

    def write_on_batch_end(self, trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx):
        output_df = self.outputs_to_dataframe(outputs)
        output_df.to_csv(self.output_file,
                         mode='a',
                         index=False,
                         header=not self.output_file.is_file())

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        output_df = pd.concat([self.outputs_to_dataframe(outputs) for outputs in predictions])
        output_df.to_csv(self.output_file,
                         mode='w',
                         index=False,
                         header=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int = 0):
        self.write_on_batch_end(trainer, pl_module, outputs, None, batch, batch_idx, dataloader_idx)

    def teardown(self, trainer, pl_module, stage: str):
        log.info(f'Predictions saved to {self.output_file}')

    @staticmethod
    def outputs_to_dataframe(prediction):
        for key, value in prediction.items():
            if isinstance(value, torch.Tensor):
                prediction[key] = value.tolist()
            else:
                prediction[key] = list(value)
        prediction_df = pd.DataFrame(prediction)
        return prediction_df
