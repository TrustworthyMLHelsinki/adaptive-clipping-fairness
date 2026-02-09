import torch

from .configurationmanager import Configuration, Hyperparameters

class OptimizerFactory:
    @staticmethod
    def get_optimizer(
            configuration: Configuration,
            hyperparams: Hyperparameters,
            model: torch.nn.Module,
        ):
        optimizer_cls = getattr(torch.optim, configuration.optimizer)
        optimizer = optimizer_cls(model.parameters(), lr=hyperparams.learning_rate)
        return optimizer

