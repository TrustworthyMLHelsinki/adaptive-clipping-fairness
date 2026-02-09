import logging
import pathlib
from typing import List

from ..configurationmanager import Configuration, Hyperparameters
from ..utils import tensor_to_python_type
from .base_callback import Callback
from .epoch_stats import RecordEpochStatsCallback
from .per_class_accuracy import RecordPerClassAccuracyCallback
from .record_losses import RecordLossesByEpochCallback, RecordTrainLossByStepCallback

log = logging.getLogger(__name__)


class CallbackHandler:
    def __init__(self, callbacks: list = []):
        self.callbacks = callbacks

    def call(self, event, *args, **kwargs):
        for callback in self.callbacks:
            event_handler = getattr(callback, event)
            event_handler(*args, **kwargs)


class CallbackFactory:
    @staticmethod
    def get_callbacks(
        configuration: Configuration, hyperparams: Hyperparameters) -> List[Callback]:

        log_dir = configuration.log_dir
        experiment_name = configuration.experiment_name
        full_log_dir = pathlib.Path(f'{log_dir}/{experiment_name}')

        callbacks = [
            RecordEpochStatsCallback(use_steps=configuration.use_steps, experiment_name=experiment_name),
        ]

        if configuration.record_loss_by_step:
            callbacks.append(RecordTrainLossByStepCallback(log_dir=full_log_dir))

        if configuration.record_loss_by_epoch:
            callbacks.append(RecordLossesByEpochCallback(log_dir=full_log_dir))

        return callbacks
