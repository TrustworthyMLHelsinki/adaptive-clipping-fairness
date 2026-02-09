import logging
import pathlib
import torch
import typer

from pydantic import BaseModel, root_validator
from typing import Optional, List, Literal

log = logging.getLogger(__name__)

class Hyperparameters(BaseModel):
    learning_rate: float = 1e-3
    epochs: Optional[int] = None
    total_steps: Optional[int] = None
    batch_size: Optional[int] = None
    noise_multiplier: Optional[float]
    max_grad_norm: Optional[float] # only for constant clipping
    target_epsilon: Optional[float]
    privacy: bool = True # Only used in __str__
    target_quantile: Optional[float]
    count_threshold: Optional[float]
    clip_bound_lr: Optional[float]
    clip_bound_lower_bound: Optional[float]
    count_noise_denom: Optional[int]
    clip_bound_init: Optional[float]

    def __str__(self):
        hypers = [
            ('Epochs', self.epochs),
            ('Total steps', self.total_steps),
            ('Learning rate', self.learning_rate),
            ('Batch size', self.batch_size),
        ]

        if self.privacy:
            privacy_hypers = [
                ('Noise multiplier', self.noise_multiplier),
                ('Clipping bound', self.max_grad_norm),
                ('Target epsilon', self.target_epsilon),
            ]
            hypers.extend(privacy_hypers)

        if self.target_quantile or self.count_threshold or self.clip_bound_lr:
            adaptive_hypers = [
                ('Target quantile', self.target_quantile),
                ('Count threshold', self.count_threshold),
                ('Clipping bound leraning rate', self.clip_bound_lr),
                ('clip_bound_lower_bound', self.clip_bound_lower_bound),
                ('count_noise_denom', self.count_noise_denom),
                ('clip_bound_init', self.clip_bound_init),
            ]
            hypers.extend(adaptive_hypers)

        max_key_length = max(len(hyper[0]) for hyper in hypers)
        hyper_str = [f'{hyper[0]:<{max_key_length}}: {hyper[1]}' for hyper in hypers]

        return 'Hyperparameters:\n  ' + '\n  '.join(hyper_str) + '\n'

class Configuration(BaseModel):
    command: Literal[
        'train',
        'optimize',
    ]
    privacy: bool = True
    model_name: str = 'resnet50'
    optimizer: str = 'Adam'
    dataset_name: str = 'adult'
    physical_batch_size: int = 40
    num_workers: int = 8
    validation_frequency: float = 1.0
    seed: int = 0
    log_dir: str = 'logs'
    experiment_name: str = 'default-experiment'
    overwrite_experiment: bool = False
    clipping_mode: str = 'flat'
    secure_mode: bool = False
    accountant: str = 'prv'
    poisson_sampling: bool = True
    normalize_clipping: bool = False
    n_trials: int = 20
    target_hypers: List[str] = []
    optuna_target_metric: str = 'loss'
    optuna_direction: Literal['minimize', 'maximize'] = 'minimize'
    optuna_config: str = 'conf/optuna_hypers.conf'
    optuna_journal: str = 'optuna.journal'
    optuna_sampler: str = 'BoTorchSampler'
    use_steps: Optional[bool] = False
    evaluation_mode: Optional[bool] = False
    dataset_label_field: Optional[str] = None
    imbalance_factor: Optional[float] = None
    fairness_imbalance_class: Optional[int] = None
    validation_size: Optional[float] = 0.1
    test_size: Optional[float] = 0.1
    verbose_callback: Optional[bool] = False
    protected_feature: Optional[str] = None
    record_loss_by_step: Optional[bool] = False
    record_loss_by_epoch: Optional[bool] = False

    class Config:
        # Fix Pydantic warning:
        # UserWarning: Field "model_name" has conflict with protected namespace "model_".
        protected_namespaces = ()

    @root_validator(pre=True)
    def check_fairness_imbalance_factor(cls, values):
        imbalance_factor = values.get('imbalance_factor')
        fairness_imbalance_class = values.get('fairness_imbalance_class')

        if fairness_imbalance_class and not imbalance_factor:
            raise ValueError(
                'Parameter "imbalance_factor" is required when using "fairness_imbalance_class".'
            )

        return values

    @root_validator(pre=True)
    def check_record_loss_by_step(cls, values):
        record_loss_by_step = values.get('record_loss_by_step')
        use_steps = values.get('use_steps')

        if record_loss_by_step and not use_steps:
            raise ValueError('Unable to record trian loss by step when using epochs. Hint: `--use-steps`')

        return values

    @root_validator(pre=True)
    def check_command(cls, values):
        command = values.get('command')

        if command not in [
            'train',
            'optimize',
        ]:
            raise ValueError('Command must be "train" or "optimize".')

        return values

    @root_validator(pre=True)
    def check_total_steps(cls, values):
        total_steps = values.get('total_steps')
        use_steps = values.get('use_steps')
        epochs = values.get('epochs')

        if total_steps and epochs:
            raise ValueError('Parameters "epochs" and "total_steps" are exclusive.')

        if total_steps and not use_steps:
            raise ValueError('Parameter "total_steps" requires also "use_steps".')

        return values

    def __str__(self):
        attributes = [
            ('Command', self.command),
            ('Privacy', self.privacy),
            ('Model name', self.model_name),
            ('Optimizer', self.optimizer),
            ('Dataset name', self.dataset_name),
            ('Dataset label field', self.dataset_label_field),
            ('Dataset imbalance factor', self.imbalance_factor),
            ('Validation size', self.validation_size),
            ('Test size', self.test_size),
            ('Physical batch size', self.physical_batch_size),
            ('Num workers', self.num_workers),
            ('Validation frequency', self.validation_frequency),
            ('Seed', self.seed),
            ('Log dir', self.log_dir),
            ('Experiment dame', self.experiment_name),
            ('Overwrite experiment', self.overwrite_experiment),
            ('Use steps instead of epochs', self.use_steps),
            ('Evaluation mode', self.evaluation_mode),
            ('Record train loss by step', self.record_loss_by_step),
            ('Record train/valid loss by epoch', self.record_loss_by_epoch),
            ('Enable the debug callback output',self.verbose_callback),
            ('Protected feature', self.protected_feature),
        ]

        if self.privacy:
            privacy_attributes = [
                ('Clipping mode', self.clipping_mode),
                ('Secure mode', self.secure_mode),
                ('Accountant', self.accountant),
                ('Poisson sampling', self.poisson_sampling),
                ('Normalize clipping', self.normalize_clipping),
            ]
            attributes.extend(privacy_attributes)

        if self.command == 'optimize':
            optuna_attributes = [
                ('N trials', self.n_trials),
                ('Target hypers', ', '.join(self.target_hypers)),
                ('Optuna target metric', self.optuna_target_metric),
                ('Optuna direction', self.optuna_direction),
                ('Optuna config', self.optuna_config),
                ('Optuna journal', self.optuna_journal),
            ]
            attributes.extend(optuna_attributes)

        max_key_length = max(len(attr[0]) for attr in attributes)
        attribute_str = [f'{attr[0]:<{max_key_length}}: {attr[1]}' for attr in attributes]

        return 'Configuration:\n  ' + '\n  '.join(attribute_str) + '\n'

class ConfigurationManager:
    def __init__(self, cli_params: dict):
        self.command = cli_params['command']

        self.configuration = Configuration(**cli_params)
        self.hyperparams = Hyperparameters(**cli_params)

        # Opacus calculates noise multiplier is target epsilon is given
        if self.hyperparams.target_epsilon is not None:
            if torch.distributed.get_rank() == 0:
                log.info('We have "target_epsilon" defined. Removing "noise_multiplier".')

            self.hyperparams.noise_multiplier = None

    def get_command(self):
        return self.command

    def save_configuration(self, directory: pathlib.Path):
        if torch.distributed.get_rank() == 0:
            with open(directory / 'configuration.txt', 'w') as fh:
                fh.write(str(self.configuration))

            with open(directory / 'configuration.json', 'w') as fh:
                fh.write(self.configuration.json())

            log.info(f'Configuration saved to {directory}.')

    def save_hyperparameters(self, directory: pathlib.Path):
        if torch.distributed.get_rank() == 0:
            with open(directory / 'hyperparameters.txt', 'w') as fh:
                fh.write(str(self.hyperparams))

            with open(directory / 'hyperparameters.json', 'w') as fh:
                fh.write(self.hyperparams.json())

            log.info(f'Hyperparameters saved to {directory}/.')
