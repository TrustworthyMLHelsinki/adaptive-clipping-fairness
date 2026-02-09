import csv
import os
import torch
import numpy as np
from .base_callback import Callback
import logging

log = logging.getLogger(__name__)


class RecordBodyAndHeadGradientNormsPerClassCallback(Callback):
    def __init__(self, log_dir: str, max_grad_norm: float):
        self.log_dir = log_dir
        self.max_grad_norm = max_grad_norm
        self.grad_history = []

    def on_train_start(self, trainer, *args, **kwargs):
        # Separate body and head of the model
        self.body, self.head = self._get_body_and_head(trainer._unwrap_model())
        self.num_classes = trainer.datamodule.get_num_classes()

    def on_train_batch_start(self, *args, **kwargs):
        # Initialize accumulators for each class at the start of each logical batch
        self.body_norms_accumulator = {cls: [] for cls in range(self.num_classes)}
        self.head_norms_accumulator = {cls: [] for cls in range(self.num_classes)}

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, *args, **kwargs):
        batch_data, class_labels = batch

        with torch.no_grad():
            for cls in range(self.num_classes):
                cls_mask = class_labels == cls
                if cls_mask.sum() == 0:
                    continue

                # Accumulate gradients from all layers
                body_grad_samples = []
                head_grad_samples = []

                for module in self.body.modules():
                    if (
                        hasattr(module, 'weight')
                        and module.weight is not None
                        and module.weight.requires_grad
                    ):
                        grad_sample = module.weight.grad_sample[cls_mask]
                        body_grad_samples.append(grad_sample.view(grad_sample.size(0), -1))

                for module in self.head.modules():
                    if (
                        hasattr(module, 'weight')
                        and module.weight is not None
                        and module.weight.requires_grad
                    ):
                        grad_sample = module.weight.grad_sample[cls_mask]
                        head_grad_samples.append(grad_sample.view(grad_sample.size(0), -1))

                # Compute the norms of gradients from all layers
                if body_grad_samples:
                    body_grad_samples = torch.cat(body_grad_samples, dim=1)
                    body_norms = body_grad_samples.norm(p=2, dim=1).cpu().numpy()
                    self.body_norms_accumulator[cls].extend(body_norms.tolist())

                if head_grad_samples:
                    head_grad_samples = torch.cat(head_grad_samples, dim=1)
                    head_norms = head_grad_samples.norm(p=2, dim=1).cpu().numpy()
                    self.head_norms_accumulator[cls].extend(head_norms.tolist())

                # Clear memory after each physical batch
                del body_grad_samples, head_grad_samples
                torch.cuda.empty_cache()

    def on_train_batch_end(self, trainer, batch_idx, *args, **kwargs):
        # Aggregate the per-class gradient norms for the logical batch
        row_data = {'step': batch_idx}

        for cls in range(self.num_classes):
            # Calculate statistics of body and head norms across physical batches
            if self.body_norms_accumulator[cls]:
                body_mean_norm = np.mean(self.body_norms_accumulator[cls])
                body_median_norm = np.median(self.body_norms_accumulator[cls])
                body_q1_norm = np.percentile(self.body_norms_accumulator[cls], 25)
                body_q3_norm = np.percentile(self.body_norms_accumulator[cls], 75)
                body_q01_norm = np.percentile(self.body_norms_accumulator[cls], 1)
                body_q10_norm = np.percentile(self.body_norms_accumulator[cls], 10)
                body_q90_norm = np.percentile(self.body_norms_accumulator[cls], 90)
                body_q99_norm = np.percentile(self.body_norms_accumulator[cls], 99)
            else:
                body_mean_norm = body_median_norm = body_q1_norm = body_q3_norm = 0.0
                body_q01_norm = body_q10_norm = body_q90_norm = body_q99_norm = 0.0

            if self.head_norms_accumulator[cls]:
                head_mean_norm = np.mean(self.head_norms_accumulator[cls])
                head_median_norm = np.median(self.head_norms_accumulator[cls])
                head_q1_norm = np.percentile(self.head_norms_accumulator[cls], 25)
                head_q3_norm = np.percentile(self.head_norms_accumulator[cls], 75)
                head_q01_norm = np.percentile(self.head_norms_accumulator[cls], 1)
                head_q10_norm = np.percentile(self.head_norms_accumulator[cls], 10)
                head_q90_norm = np.percentile(self.head_norms_accumulator[cls], 90)
                head_q99_norm = np.percentile(self.head_norms_accumulator[cls], 99)
            else:
                head_mean_norm = head_median_norm = head_q1_norm = head_q3_norm = 0.0
                head_q01_norm = head_q10_norm = head_q90_norm = head_q99_norm = 0.0

            # Store the calculated norms
            row_data[f'Class_{cls}_Body_Mean_Norm'] = body_mean_norm
            row_data[f'Class_{cls}_Body_Median_Norm'] = body_median_norm
            row_data[f'Class_{cls}_Body_Q1_Norm'] = body_q1_norm
            row_data[f'Class_{cls}_Body_Q3_Norm'] = body_q3_norm
            row_data[f'Class_{cls}_Body_Q01_Norm'] = body_q01_norm
            row_data[f'Class_{cls}_Body_Q10_Norm'] = body_q10_norm
            row_data[f'Class_{cls}_Body_Q90_Norm'] = body_q90_norm
            row_data[f'Class_{cls}_Body_Q99_Norm'] = body_q99_norm
            row_data[f'Class_{cls}_Head_Mean_Norm'] = head_mean_norm
            row_data[f'Class_{cls}_Head_Median_Norm'] = head_median_norm
            row_data[f'Class_{cls}_Head_Q1_Norm'] = head_q1_norm
            row_data[f'Class_{cls}_Head_Q3_Norm'] = head_q3_norm
            row_data[f'Class_{cls}_Head_Q01_Norm'] = head_q01_norm
            row_data[f'Class_{cls}_Head_Q10_Norm'] = head_q10_norm
            row_data[f'Class_{cls}_Head_Q90_Norm'] = head_q90_norm
            row_data[f'Class_{cls}_Head_Q99_Norm'] = head_q99_norm

        # Save the data for this batch
        self.grad_history.append(row_data)

        # Clear accumulators for the next logical batch
        for cls in range(self.num_classes):
            self.body_norms_accumulator[cls].clear()
            self.head_norms_accumulator[cls].clear()

    def on_train_end(self, trainer, *args, **kwargs):
        if self._is_global_zero():
            file_path = os.path.join(
                self.log_dir, 'gradient_norms_body_head_per_class.csv'
            )
            self._save_to_csv(file_path)

    def _get_body_and_head(self, model):
        head = model.get_classifier()
        body = model.get_body()

        return body, head


    def _save_to_csv(self, file_path):
        fieldnames = ['step']
        fieldnames += [f'Class_{cls}_Body_Mean_Norm' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Body_Median_Norm' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Body_Q1_Norm' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Body_Q3_Norm' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Body_Q01_Norm' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Body_Q10_Norm' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Body_Q90_Norm' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Body_Q99_Norm' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Head_Mean_Norm' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Head_Median_Norm' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Head_Q1_Norm' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Head_Q3_Norm' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Head_Q01_Norm' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Head_Q10_Norm' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Head_Q90_Norm' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Head_Q99_Norm' for cls in range(self.num_classes)]

        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.grad_history)

        log.info(f'Gradient norms (body & head per class) saved to {file_path}')

