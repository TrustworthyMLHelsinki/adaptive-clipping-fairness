import logging
import timm
import torch

import torch.nn as nn

from .model_base import ModelBase
from .koskela_model import KoskelaNetGrayscale
from .simple_nn import MisAlignNN
from .logistic_model import LogisticRegression

from dpdl.configurationmanager import Configuration, Hyperparameters

from opacus.validators import ModuleValidator

log = logging.getLogger(__name__)

def add_noise_to_weights(model, noise_level):
    for name, param in model.named_parameters():
        if 'weight' in name:
            noise = torch.randn(param.size()) * noise_level
            param.data.add_(noise)

class ModelFactory:

    @staticmethod
    def get_model(
        configuration: Configuration,
        hyperparams: Hyperparameters,
        num_classes: int,

    ):
        """
        Create a model instance based on the configuration.

        Parameters:
        - configuration: Configuration object containing model specs.
        - hyperparams: Optional hyperparameters, not directly used here.
        - num_classes: The number of classes for a classification problem

        Returns:
        - A tuple of (ModelBase instance, Data Transforms).
        """

        transforms = {}  # No default transforms
        model_instance = None

        if configuration.model_name == 'koskela-net':
            model_instance = KoskelaNet()
            transforms = model_instance.get_transforms()
        elif configuration.model_name == 'koskela-net-grayscale':
            model_instance = KoskelaNetGrayscale()
            transforms = model_instance.get_transforms()
        elif configuration.model_name == 'simple-nn':
            model_instance = MisAlignNN()
            transforms = model_instance.get_transforms()
        elif configuration.model_name == 'logistic':
            imput_dim = 96 if configuration.dataset_name == 'adult' else 50
            model_instance = LogisticRegression(imput_dim)
            transforms = model_instance.get_transforms()
            # Resolve data config and create transforms
            #model_config = timm.data.resolve_data_config({}, model=model_instance.model)
            model_config = {'input_size': (1, 224, 224),'mean': [0.5], 'std': [0.5]}
            transforms = timm.data.transforms_factory.create_transform(**model_config)

        # Wrap the instantiated model with ModelBase
        model = ModelBase(
            model_instance=model_instance,
            num_classes=num_classes,
        )

        if 'resnet' in configuration.model_name:
            log.info("Model is fixed with ModuleValidator.fix")
            model = ModuleValidator.fix(model)

        return model, transforms
