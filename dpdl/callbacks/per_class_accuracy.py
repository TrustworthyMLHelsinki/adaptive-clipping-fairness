import csv
import os
import torch
from .base_callback import Callback
import logging

log = logging.getLogger(__name__)


class RecordPerClassAccuracyCallback(Callback):
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.per_class_accuracies_history = []

    def on_train_batch_end(self, trainer, *args, **kwargs):
        # At the end of the logical batch, we compute and save per-class accuracies
        train_metrics = trainer._unwrap_model().train_metrics.compute()

        # Extract per-class accuracies
        per_class_accuracies = train_metrics.get('MulticlassAccuracyPerClass', None)

        if per_class_accuracies is not None:
            # Convert to a list for easier logging and saving
            per_class_accuracies_list = per_class_accuracies.tolist()

            self.per_class_accuracies_history.append(per_class_accuracies_list)

    def on_train_end(self, trainer, *args, **kwargs):
        if self._is_global_zero():
            file_path = os.path.join(self.log_dir, 'per-class-accuracies.csv')

            # Save the per-class accuracies history to a CSV
            with open(file_path, 'w', newline='') as fh:
                writer = csv.writer(fh)

                # Construct header row for the CSV
                header = ['Step']

                for i in range(len(self.per_class_accuracies_history[0])):
                    header += [f'Class_{i}']

                writer.writerow(header)

                # Write each batch's per-class accuracies
                for i, accuracies in enumerate(self.per_class_accuracies_history):
                    writer.writerow([i] + accuracies)

            log.info(f'Per-class accuracy data saved at {file_path}')