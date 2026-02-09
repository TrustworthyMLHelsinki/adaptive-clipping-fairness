import math
import logging
import torch
import torchmetrics
from .base_callback import Callback
import csv
import os

log = logging.getLogger(__name__)


class RecordEpochStatsCallback(Callback):
    def __init__(self, use_steps=False, experiment_name=None):
        self.use_steps = use_steps
        self.experiment_name = experiment_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_loss = torchmetrics.aggregation.MeanMetric().to(self.device)
        self.evaluation_loss = torchmetrics.aggregation.MeanMetric(
            sync_on_compute=False
        ).to(self.device)
        self.adaptive_cb_history = []
        self.metric_history = []

    def on_train_start(self, trainer):
        if self._is_global_zero():
            if self.use_steps:
                batch_size = trainer.datamodule.batch_size
                data_size = len(trainer.get_dataloader("train").dataset)
                steps_per_epoch = data_size / batch_size
                epochs = math.ceil(trainer.total_steps / steps_per_epoch)

                log.info(
                    f"!!! Starting training for approximately {epochs} epochs ({trainer.total_steps} steps)."
                )
            else:
                log.info(f"!!! Starting training for {trainer.epochs} epochs.")

    def on_train_end(self, trainer):
        if self._is_global_zero():
            log.info("!!! Training finished.")

    def on_train_epoch_start(self, trainer, epoch):
        self.train_loss.reset()

        if self._is_global_zero():
            log.info(f"--------------------------------------------------")
            if not self.use_steps:
                log.info(f"Starting training epoch {epoch+1}.")
            else:
                log.info(f"Starting training approximate epoch {epoch+1}.")

    def on_train_epoch_end(self, trainer, epoch, metrics):
        loss = self.train_loss.compute()

        if self._is_global_zero():
            if not self.use_steps:
                log.info(f"Epoch {epoch+1} finished. Loss: {loss:.4f}.")
            else:
                log.info(f"Approximate epoch {epoch+1} finished. Loss: {loss:.4f}.")
            self._log_metrics(metrics, "Train metrics")

            def _convert_tensor(item):
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu()
                    return item.item() if item.dim() == 0 else item.tolist()
                return item

            os.makedirs("cb_history", exist_ok=True)
            clipbound = getattr(trainer.optimizer, "clipbound", None)
            log.info(f" - clipbound: {clipbound}")

            row_data = {"epoch": epoch + 1, "clipbound": _convert_tensor(clipbound)}
            for k, v in metrics.items():
                row_data[k] = _convert_tensor(v)

            csv_path = os.path.join("cb_history", f"{self.experiment_name}.csv")
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)
            pass

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        self.train_loss.update(loss)

    def on_validation_epoch_end(self, trainer, epoch, metrics):
        loss = self.evaluation_loss.compute()
        self.evaluation_loss.reset()

        if self._is_global_zero():
            log.info(f"Validation finished. Loss: {loss:.4f}.")
            self._log_metrics(metrics, "Validation metrics")

    def on_validation_batch_end(self, trainer, batch_idx, batch, loss):
        self.evaluation_loss.update(loss)

    def on_test_epoch_end(self, trainer, epoch, metrics):
        loss = self.evaluation_loss.compute()
        self.evaluation_loss.reset()

        if self._is_global_zero():
            log.info(f"Test finished. Loss: {loss:.4f}.")
            self._log_metrics(metrics, "Test metrics")

    def on_test_batch_end(self, trainer, batch_idx, batch, loss):
        self.evaluation_loss.update(loss)
