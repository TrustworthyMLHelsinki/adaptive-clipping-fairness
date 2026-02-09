import datetime
import json
import logging
import pathlib
import pickle
import shutil
import subprocess

import optuna
import torch

from .configurationmanager import ConfigurationManager
from .trainer import Trainer
from .utils import tensor_to_python_type

log = logging.getLogger(__name__)

def save_study(
        config_manager: ConfigurationManager,
        study: optuna.study.Study,
        final_metrics: dict,
    ):

    # unwrap metric  values from torch tensors
    final_metrics = tensor_to_python_type(final_metrics)

    log_dir = config_manager.configuration.log_dir
    experiment_name = config_manager.configuration.experiment_name
    full_log_dir = pathlib.Path(f'{log_dir}/{experiment_name}')

    with open(full_log_dir / 'trials.json', 'w') as fh:
        fh.write(study.trials_dataframe().to_json())

    with open(full_log_dir / 'trials.csv', 'w') as fh:
        fh.write(study.trials_dataframe().to_csv())

    with open(full_log_dir / 'best-params.json', 'w') as fh:
        json.dump(study.best_params, fh)

    with open(full_log_dir / 'best-value', 'w') as fh:
        fh.write(str(study.best_value) + '\n')

    with open(full_log_dir / 'final-metrics', 'w') as fh:
        json.dump(final_metrics, fh)

    with open(full_log_dir / 'results-and-configuration.json', 'w') as fh:
        d = {}
        d['best_params'] = study.best_params
        d['best_value'] = study.best_value
        d['configuration'] = config_manager.configuration.dict()
        d['final_metrics'] = final_metrics
        d['hyperparameters'] = config_manager.hyperparams.dict()

        json.dump(d, fh)

    _copy_optuna_study_to_experiment_dir(config_manager)

    _copy_optuna_config_to_experiment_dir(config_manager)

def _copy_optuna_config_to_experiment_dir(config_manager: ConfigurationManager):
    src_path = config_manager.configuration.optuna_config

    log_dir = config_manager.configuration.log_dir
    experiment_name = config_manager.configuration.experiment_name
    dst_path = pathlib.Path(f'{log_dir}/{experiment_name}/optuna.conf')

    shutil.copy(src_path, dst_path)

def _copy_optuna_study_to_experiment_dir(config_manager: ConfigurationManager):
    experiment_name = config_manager.configuration.experiment_name

    # source storage is the main optuna journal file
    src_journal_fpath = str(config_manager.configuration.optuna_journal) # optuna expects strings
    src_storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(src_journal_fpath))

    # destination storage is under the experiment directory
    log_dir = config_manager.configuration.log_dir
    experiment_name = config_manager.configuration.experiment_name
    dst_journal_fpath = str(pathlib.Path(f'{log_dir}/{experiment_name}/optuna.journal'))
    dst_storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(dst_journal_fpath))

    # now copy this experiment's journal to the experiment directory
    optuna.copy_study(
        from_study_name=experiment_name,
        from_storage=src_storage,
        to_storage=dst_storage,
    )

def save_hpo_metrics(
        config_manager: ConfigurationManager,
        loss: float,
        metrics: dict,
        trial_index: int,
    ):
    log_dir = config_manager.configuration.log_dir
    experiment_name = config_manager.configuration.experiment_name
    full_log_dir = pathlib.Path(f'{log_dir}/{experiment_name}')

    metrics = tensor_to_python_type(metrics)

    # if exists, read the data
    if (full_log_dir / 'hpo_metrics.json').exists():
        with open(full_log_dir / 'hpo_metrics.json', 'r') as fh:
            data = json.load(fh)
    else:
        data = []

    data.append({
        'trial_index': trial_index,
        'loss': loss,
        **metrics,
    })

    # save the data
    with open(full_log_dir / 'hpo_metrics.json', 'w') as fh:
        json.dump(data, fh)


def start_experiment_logging(
        log: logging.Logger,
        config_manager: ConfigurationManager,
        overwrite: bool = False,
    ):

    log_dir = config_manager.configuration.log_dir
    experiment_name = config_manager.configuration.experiment_name

    # create a directory for the experiments and start logging there
    overwrite = config_manager.configuration.overwrite_experiment
    experiment_directory = _create_experiment_directory(log_dir, experiment_name, overwrite)
    _start_logging_to_files(log, experiment_directory)

    # save configuration
    config_manager.save_configuration(experiment_directory)
    config_manager.save_hyperparameters(experiment_directory)

    # if ovewriting, delete the possible existing study from optuna journal
    if config_manager.configuration.overwrite_experiment:
        _delete_optuna_study(config_manager)

    # log the gpu type and count
    _log_gpus(config_manager)

    _log_git_hash(config_manager)

def _log_git_hash(config_manager):
    log_dir = config_manager.configuration.log_dir
    experiment_name = config_manager.configuration.experiment_name
    full_log_dir = pathlib.Path(f'{log_dir}/{experiment_name}')

    with open(full_log_dir / 'git-hash', 'w') as fh:
        git_hash = _get_git_hash()
        fh.write(str(git_hash) + '\n')

def log_runtime(config_manager, start_time, end_time):
    elapsed = end_time - start_time
    elapsed_timedelta = datetime.timedelta(seconds=elapsed)

    log_dir = config_manager.configuration.log_dir
    experiment_name = config_manager.configuration.experiment_name
    full_log_dir = pathlib.Path(f'{log_dir}/{experiment_name}')

    with open(f'{full_log_dir}/runtime', 'w') as fh:
        fh.write(f'{elapsed_timedelta}\n')

def log_test_metrics(config_manager, metrics, loss):
    log_dir = config_manager.configuration.log_dir
    experiment_name = config_manager.configuration.experiment_name
    full_log_dir = pathlib.Path(f'{log_dir}/{experiment_name}')

    metrics = tensor_to_python_type(metrics)
    metrics['loss'] = loss

    with open(f'{full_log_dir}/test_metrics', 'w') as fh:
        json.dump(metrics, fh)

def log_final_epsilon(config_manager, trainer):
    if not config_manager.configuration.privacy:
        return

    if config_manager.hyperparams.target_epsilon == -1:
        return

    log_dir = config_manager.configuration.log_dir
    experiment_name = config_manager.configuration.experiment_name
    full_log_dir = pathlib.Path(f'{log_dir}/{experiment_name}')

    final_epsilon = trainer.get_epsilon()
    with open(f'{full_log_dir}/final_epsilon', 'w') as fh:
        fh.write(f'{final_epsilon}\n')

def _log_gpus(config_manager):
    log_dir = config_manager.configuration.log_dir
    experiment_name = config_manager.configuration.experiment_name
    full_log_dir = pathlib.Path(f'{log_dir}/{experiment_name}')

    with open(f'{full_log_dir}/gpu_type', 'w') as fh:
        gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
        fh.write(f'{gpu_name}\n')

    with open(f'{full_log_dir}/gpu_count', 'w') as fh:
        gpu_count = torch.distributed.get_world_size()
        fh.write(f'{gpu_count}\n')

def _delete_optuna_study(config_manager: ConfigurationManager):
    # storage is the main optuna journal file
    journal_fpath = str(config_manager.configuration.optuna_journal) # optuna expects strings
    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(journal_fpath))

    experiment_name = config_manager.configuration.experiment_name

    try:
        optuna.delete_study(study_name=experiment_name, storage=storage)
    except KeyError:
        # study did not exist
        pass

def _get_git_hash():
    process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    git_hash = process.communicate()[0].strip()
    return git_hash

def _create_experiment_directory(
        log_dir: str = 'log_dir',
        experiment_name: str = 'Default experiment',
        overwrite: bool = False,
    ) -> pathlib.Path:

    full_log_dir = pathlib.Path(f'{log_dir}/{experiment_name}')

    if full_log_dir.exists() and overwrite:
        log.info(f'Experiment directory "{full_log_dir}" exists, removing it and restarting experiment.')
        shutil.rmtree(full_log_dir)

    if full_log_dir.exists() and not overwrite:
        log.info(f'Experiment directory "{full_log_dir}" exists, resuming experiment.')

    if not full_log_dir.exists():
        log.info(f'Creating experiment directory "{full_log_dir}".')
        full_log_dir.mkdir(parents=True)

    return full_log_dir

def _start_logging_to_files(log: logging.Logger, log_path: pathlib.Path):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create a file handler for saving stdout logs to a file
    stdout_file_handler = logging.FileHandler(log_path / 'stdout.txt')
    stdout_file_handler.setLevel(logging.INFO)
    stdout_file_handler.setFormatter(formatter)
    log.addHandler(stdout_file_handler)

    # create a file handler for saving stderr logs to a file
    stderr_file_handler = logging.FileHandler(log_path / 'stderr.txt')
    stderr_file_handler.setLevel(logging.WARNING)
    stderr_file_handler.setFormatter(formatter)
    log.addHandler(stderr_file_handler)
