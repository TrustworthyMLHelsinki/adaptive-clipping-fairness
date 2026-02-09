import logging
import torch

from ..utils import tensor_to_python_type

log = logging.getLogger(__name__)


class Callback:
    def _is_global_zero(self):
        return torch.distributed.get_rank() == 0

    def on_train_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_train_epoch_start(self, trainer, epoch):
        pass

    def on_train_epoch_end(self, trainer, epoch, epoch_loss):
        pass

    def on_train_batch_start(self, trainer, batch_idx, batch):
        pass

    def on_train_physical_batch_start(self, trainer, batch_idx, batch):
        pass

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        pass

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, loss):
        pass

    def on_validation_epoch_start(self, trainer, epoch):
        pass

    def on_validation_epoch_end(self, trainer, epoch, metrics):
        pass

    def on_validation_batch_start(self, trainer, batch_idx, batch):
        pass

    def on_validation_batch_end(self, trainer, batch_idx, batch, loss):
        pass

    def on_test_epoch_start(self, trainer, epoch):
        pass

    def on_test_epoch_end(self, trainer, epoch, metrics):
        pass

    def on_test_batch_start(self, trainer, batch_idx, batch):
        pass

    def on_test_batch_end(self, trainer, batch_idx, batch, loss):
        pass

    def _log_metrics(self, metrics, annotation="Metrics"):
        if not metrics:
            return

        metrics = tensor_to_python_type(metrics)

        log.info(annotation + ":")
        for key, value in metrics.items():
            log.info(f" - {key}: {value}.")
