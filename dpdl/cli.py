import logging
import time
import torch
import typer
import sys

from typing import Optional, List
from typing_extensions import Annotated

from .configurationmanager import ConfigurationManager
from .trainer import TrainerFactory
from .hyperparameteroptimizer import HyperparameterOptimizer
from .utils import seed_everything
from .experimentmanager import start_experiment_logging, log_runtime, log_test_metrics, log_final_epsilon

log = logging.getLogger(__name__)


def cli(
    ctx: typer.Context,
    command: Annotated[
        str,
        typer.Argument(
            help='Command to run ("train", "optimize", or "show-layers")',
        ),
    ],
    use_steps: Annotated[
        bool,
        typer.Option(
            help='Use steps instead of epochs',
            rich_help_panel='Training options',
        ),
    ] = False,
    epochs: Annotated[
        int,
        typer.Option(
            help='Number of epochs to train',
            rich_help_panel='Training options',
        ),
    ] = None,
    total_steps: Annotated[
        int,
        typer.Option(
            help='Total number of gradient updates',
            rich_help_panel='Training options',
        ),
    ] = None,
    learning_rate: Annotated[
        float,
        typer.Option(
            help='Learning rate',
            rich_help_panel='Training options',
        ),
    ] = 1,
    batch_size: Annotated[
        Optional[int],
        typer.Option(
            help='Batch size',
            rich_help_panel='Training options',
        ),
    ] = None,
    optimizer: Annotated[
        str,
        typer.Option(
            help='Optimizer',
            rich_help_panel='Training options',
        ),
    ] = 'SGD',
    physical_batch_size: Annotated[
        Optional[int],
        typer.Option(
            help='Largest size batch that fits in GPU memory',
            rich_help_panel='Training options',
        ),
    ] = 40,
    num_workers: Annotated[
        int,
        typer.Option(
            help='Number of workers for data loading (per GPU)',
            rich_help_panel='Training options',
        ),
    ] = 8,
    validation_frequency: Annotated[
        float,
        typer.Option(
            help='Validation frequency',
            rich_help_panel='Training options',
        ),
    ] = 1.0,
    seed: Annotated[
        int,
        typer.Option(
            help='Random seed',
            rich_help_panel='Training options',
        ),
    ] = 0,
    privacy: Annotated[
        bool,
        typer.Option(
            help='Enable privacy (Opacus)',
            rich_help_panel='Training options',
        ),
    ] = True,
    evaluation_mode: Annotated[
        bool,
        typer.Option(
            help='Enable evaluation mode (train on train+valid and validate on test)',
            rich_help_panel='Training options',
        ),
    ] = False,
    model_name: Annotated[
        str,
        typer.Option(
            help='PyTorch Image Models (timm) model name',
            rich_help_panel='Model options',
        ),
    ] = 'koskela-net',
    dataset_name: Annotated[
        str,
        typer.Option(
            help='Dataset name',
            rich_help_panel='Dataset options',
        ),
    ] = 'adult',
    validation_size: Annotated[
        Optional[float],
        typer.Option(
            help='Validation set size, if we need to split it from train (0.1 indicates 10%)',
            rich_help_panel='Dataset options',
        ),
    ] = 0.1,
    test_size: Annotated[
        Optional[float],
        typer.Option(
            help='Test set size, if we need to split it from train (0.1 indicates 10%)',
            rich_help_panel='Dataset options',
        ),
    ] = 0.1,
    dataset_label_field: Annotated[
        Optional[str],
        typer.Option(
            help='Name of the field that determines label for the dataset',
            rich_help_panel='Dataset options',
        ),
    ] = None,
    imbalance_factor: Annotated[
        Optional[float],
        typer.Option(
            help='Parameter of the exponential distribution for imbalancing an dataset',
            rich_help_panel='Dataset options',
        ),
    ] = None,
    fairness_imbalance_class: Annotated[
        Optional[int],
        typer.Option(
            help="Class to imbalance for fairness experiments",
            rich_help_panel="Dataset options",
        ),
    ] = None,
    log_dir: Annotated[
        str,
        typer.Option(
            help='Log directory',
            rich_help_panel='Logging options',
        ),
    ] = 'logs',
    experiment_name: Annotated[
        Optional[str],
        typer.Option(
            help='Experiment name for logging',
            rich_help_panel='Logging options',
        ),
    ] = 'default',
    overwrite_experiment: Annotated[
        bool,
        typer.Option(
            help='Overwrite existing experiment logs',
            rich_help_panel='Logging options',
        ),
    ] = False,
    record_loss_by_step: Annotated[
        Optional[bool],
        typer.Option(
            help='Record train loss by step',
            rich_help_panel='Logging options',
        ),
    ] = False,
    record_loss_by_epoch: Annotated[
        Optional[bool],
        typer.Option(
            help='Record train/validation loss by epoch',
            rich_help_panel='Logging options',
        ),
    ] = False,
    noise_multiplier: Annotated[
        Optional[float],
        typer.Option(
            help='Noise multiplier',
            rich_help_panel='Opacus options',
        ),
    ] = None,
    max_grad_norm: Annotated[
        Optional[float],
        typer.Option(
            help='clipping bound, a.k.a., maximum gradient norm (for constant clipping)',
            rich_help_panel='Opacus options',
        ),
    ] = 1.0,
    target_quantile: Annotated[
        Optional[float],
        typer.Option(
            help='Target quantile for the adaptive clipping (for "adaptive" clipping mode)',
            rich_help_panel='Opacus options',
        ),
    ] = 1.0,
    count_threshold: Annotated[
        Optional[float],
        typer.Option(
            help='Count threshold for the adaptive clipping (for "adaptive" clipping mode)',
            rich_help_panel='Opacus options',
        ),
    ] = 1.0,
    clip_bound_lr: Annotated[
        Optional[float],
        typer.Option(
            help='Clip bound learning rate for the adaptive clipping (for "adaptive" clipping mode)',
            rich_help_panel='Opacus options',
        ),
    ] = 1.0,
    clip_bound_lower_bound: Annotated[
        Optional[float],
        typer.Option(
            help='The lower bound of adaptive clipping bound (for "adaptive" clipping mode)',
            rich_help_panel='Hyperparameter options',
        ),
    ] = None,
    count_noise_denom: Annotated[
        Optional[int],
        typer.Option(
            help='Count noise denominator for the adaptive clipping (for "adaptive" clipping mode)',
            rich_help_panel='Opacus options',
        ),
    ] = 10,
    clip_bound_init: Annotated[
        Optional[float],
        typer.Option(
            help='Initial clipping bound (for "adaptive" clipping mode)',
            rich_help_panel='Hyperparameter options',
        ),
    ] = 1.0,
    clipping_mode: Annotated[
        Optional[str],
        typer.Option(
            help='Opacus clipping mode ("flat" or "per_layer" or "adaptive" or "auto")',
            rich_help_panel='Opacus options',
        ),
    ] = 'flat',
    secure_mode: Annotated[
        Optional[bool],
        typer.Option(
            help='Enable secure mode for production use',
            rich_help_panel='Opacus options',
        ),
    ] = False,
    poisson_sampling: Annotated[
        Optional[bool],
        typer.Option(
            help='Enable Opacus Poisson sampling',
            rich_help_panel='Opacus options',
        ),
    ] = True,
    normalize_clipping: Annotated[
        Optional[bool],
        typer.Option(
            help='Normalize clipping (to decouple the learning rate and max_grad_norm)',
            rich_help_panel='Opacus options',
        ),
    ] = False,
    accountant: Annotated[
        Optional[str],
        typer.Option(
            help='Privacy accountant',
            rich_help_panel='Opacus options',
        ),
    ] = 'prv',
    target_epsilon: Annotated[
        Optional[float],
        typer.Option(
            help='Target epsilon for the privacy accountant (implies delta = 1/N)',
            rich_help_panel='Opacus options',
        ),
    ] = None,
    target_hypers: Annotated[
        Optional[List[str]],
        typer.Option(
            help='Hyperparameters to optimize (use multiple times if necessary)',
            rich_help_panel='Bayesian optimization (Optuna) options',
        ),
    ] = [],
    n_trials: Annotated[
        Optional[int],
        typer.Option(
            help='Number of optimization rounds',
            rich_help_panel='Bayesian optimization (Optuna) options',
        ),
    ] = 20,
    optuna_target_metric: Annotated[
        Optional[str],
        typer.Option(
            help='Target metric for Bayesian optimization',
            rich_help_panel='Bayesian optimization (Optuna) options',
        ),
    ] = 'loss',
    optuna_direction: Annotated[
        Optional[str],
        typer.Option(
            help='Direction for Bayesian optimization ("minimize" or "maximize")',
            rich_help_panel='Bayesian optimization (Optuna) options',
        ),
    ] = 'minimize',
    optuna_config: Annotated[
        Optional[str],
        typer.Option(
            help='Configuration file containing ranges/options for hypers',
            rich_help_panel='Bayesian optimization (Optuna) options',
        ),
    ] = 'conf/optuna_hypers.conf',
    optuna_journal: Annotated[
        Optional[str],
        typer.Option(
            help='Optuna journal (logging) file path',
            rich_help_panel='Bayesian optimization (Optuna) options',
        ),
    ] = 'optuna.journal',
    optuna_sampler: Annotated[
        Optional[str],
        typer.Option(
            help='Optuna sampler (a class from optuna.samplers)',
            rich_help_panel='Bayesian optimization (Optuna) options',
        ),
    ] = 'RandomSampler',
    verbose_callback: Annotated[
        Optional[bool],
        typer.Option(
            help='Enable debug callback for detailed output',
            rich_help_panel='',
        ),
    ] = False,
    protected_feature: Annotated[
        Optional[str],
        typer.Option(
            help='Protected feature for fairness experiments',
            rich_help_panel='Fairness options',
        ),
    ] = None,
):

    config_manager = ConfigurationManager(ctx.params)
    # ConfigurationManager knows our experiment directory, so let's start logging also there
    if torch.distributed.get_rank() == 0:
        start_experiment_logging(log.parent, config_manager)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()

    if config_manager.get_command() == 'train':
        if torch.distributed.get_rank() == 0:
            log.info('Starting training.')
            log.info(config_manager.hyperparams)
            log.info(config_manager.configuration)

        seed_everything(config_manager.configuration.seed)

        trainer = TrainerFactory.get_trainer(config_manager)

        start_time = time.time()
        trainer.fit()
        end_time = time.time()

        # log test accuracy and run time, and save model if asked
        if torch.distributed.get_rank() == 0:
            log.info('Evaluating on test set..')
            test_loss, test_metrics = trainer.test()

            log_test_metrics(config_manager, test_metrics, test_loss)
            log_runtime(config_manager, start_time, end_time)
            log_final_epsilon(config_manager, trainer)

    if config_manager.get_command() == 'optimize':
        if torch.distributed.get_rank() == 0:
            log.info('Starting hyperparameter optimization.')
            log.info(config_manager.configuration)

        seed_everything(config_manager.configuration.seed)

        start_time = time.time()
        HyperparameterOptimizer.optimize_hypers(config_manager)
        end_time = time.time()

        # log the runtime
        if torch.distributed.get_rank() == 0:
            log_runtime(config_manager, start_time, end_time)
