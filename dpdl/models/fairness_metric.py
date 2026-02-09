import torch
from torchmetrics import Metric


class GroupAccuracy(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes

        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("groups", default=[], dist_reduce_fx=None)

    def update(self, preds, targets, protected_feature):
        binary_preds = (preds > 0.5).long().squeeze()
        assert (
            binary_preds.shape == targets.shape
        ), f"Shape mismatch: preds {binary_preds.shape}, targets {targets.shape}"
        self.preds.append(binary_preds)
        self.targets.append(targets.long())
        self.groups.append(protected_feature)

    def compute(self):
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        groups = torch.cat(self.groups)

        group_accuracies = []
        for group_value in groups.unique():
            mask = groups == group_value
            group_preds = preds[mask]
            group_targets = targets[mask]
            correct = (group_preds == group_targets).sum().item()
            total = len(group_preds)
            group_accuracy = correct / total if total > 0 else 0.0
            group_accuracies.append(group_accuracy)
        return torch.tensor(group_accuracies)

    def reset(self):
        self.preds = []
        self.targets = []
        self.groups = []
