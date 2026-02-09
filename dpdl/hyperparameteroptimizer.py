import gc

import logging
import optuna
import torch
import typer
import yaml

from functools import partial

from .trainer import TrainerFactory
from .datamodules import DataModuleFactory
from .models.model_factory import ModelFactory
from .configurationmanager import ConfigurationManager
from .experimentmanager import save_study, log_final_epsilon, save_hpo_metrics

log = logging.getLogger(__name__)


class HyperparameterOptimizer:

    # helper function for reading Optuna hyperparameter configuration that
    # includes types, options, lower and upper bounds, etc for hyperparams
    @staticmethod
    def read_optuna_config(config_fpath):
        with open(config_fpath, 'rb') as fh:
            config = yaml.safe_load(fh)

        return config

    # helper function for reading possible manual trials configued
    @staticmethod
    def read_manual_trials(manual_trials_fpath):
        with open(manual_trials_fpath, 'rb') as fh:
            manual_trials = yaml.safe_load(fh)

        return manual_trials.get('trials', [])

    @staticmethod
    def validate_manual_trials(manual_trials, target_hypers):
        for trial in manual_trials:
            invalid_keys = [key for key in trial if key not in target_hypers]
            if invalid_keys:
                raise ValueError(f'Invalid hyperparameters in manual trial: {invalid_keys}. These should be defined in `--target-hypers`')

    @staticmethod
    def optimize_hypers(config_manager: ConfigurationManager) -> None:

        configuration = config_manager.configuration

        # the targeted hypers from command line
        target_hypers = configuration.target_hypers

        if len(target_hypers) == 0:
            raise typer.BadParameter('Bayesian optimization enabled and no target hyperparameters for optimization.')

        optuna_config = HyperparameterOptimizer.read_optuna_config(configuration.optuna_config)

        # check that the target hypers are known and appear in Optuna configuration
        for target_hyper in target_hypers:
            if not hasattr(config_manager.hyperparams, target_hyper):
                raise typer.BadParameter(f'Cannot optimize unknown hyperparameter "{target_hyper}".')
            if not target_hyper in optuna_config:
                config_fpath = configuration.optuna_config
                raise typer.BadParameter(f'Hyperparameter "{target_hyper}" not found in Optuna configuration file "{config_fpath}".')

        # optuna needs a global process group with 'gloo' backend for communication.
        # "Create a global gloo backend when group is None and WORLD is nccl."
        # # - https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.TorchDistributedTrial.html
        optuna_process_group = torch.distributed.new_group(
            backend='gloo',
        )

        if torch.distributed.get_rank() == 0:
            log.info('Determining maximum batch size for optimization.')

        max_batch_size = HyperparameterOptimizer.get_max_batch_size(config_manager)

        if torch.distributed.get_rank() == 0:
            log.info(f'- Maximum batch size for optimization: {max_batch_size}.')

        # the optimization objective
        objective = partial(
            HyperparameterOptimizer.objective,
            config_manager,
            optuna_config,
            target_hypers,
            max_batch_size,
            optuna_process_group,
        )

        # start Optuna study only on rank zero
        if torch.distributed.get_rank() == 0:
            journal_fpath = configuration.optuna_journal

            # we will store the information about the trials on disk in a journal file
            storage = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(journal_fpath),
            )

            # we manually define the sampler to be able to set the seed
            if configuration.optuna_sampler == 'BoTorchSampler':
                sampler_cls = getattr(optuna.integration, configuration.optuna_sampler)
                sampler = sampler_cls(seed=configuration.seed)
            else:
                sampler_cls = getattr(optuna.samplers, configuration.optuna_sampler)
                sampler = sampler_cls(seed=configuration.seed)

            study = optuna.create_study(
                storage=storage,
                study_name=configuration.experiment_name,
                sampler=sampler,
                direction=configuration.optuna_direction,
            )

            study.optimize(
                objective,
                n_trials=configuration.n_trials,
                gc_after_trial=True, # garbage collect after each trial
            )
        else:
            # "Please set trial object in rank-0 node and set `None` in the other rank node.
            # - https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.TorchDistributedTrial.html
            for _ in range(configuration.n_trials):
                objective(None)

        # log the results of the best trial
        if torch.distributed.get_rank() == 0:
            trial = study.best_trial
            log.info(f'Best objective ralue: {trial.value}')
            log.info('Params: ')
            for key, value in trial.params.items():
                log.info(f' - {key}: {value}')

        # first we need to broadcast the best parameters to all ranks,
        # so we pack them into a list for sending
        if torch.distributed.get_rank() == 0:
            # rank 0 is the source
            best_params = study.best_trial.params
            broadcast_objects = [best_params]
        else:
            # other ranks receive
            broadcast_objects = [None]

        # now, broadcast the list from rank 0 to all the other ranks
        torch.distributed.broadcast_object_list(broadcast_objects, src=0)

        # and finally, let's unpack the best params from the list
        best_params = broadcast_objects[0]

        # now we can train/evaluate for the final time with best params
        metrics = HyperparameterOptimizer._final_evaluation_round(best_params, config_manager)

        if torch.distributed.get_rank() == 0:
            # save this study to experiment directory
            save_study(config_manager, study, metrics)

    @staticmethod
    def _final_evaluation_round(best_params, config_manager):
        # lastly, we'll train with the training data and the validation data
        # using the best params to get a model that we can evaluate on the test
        # for the final accuracy set
        if torch.distributed.get_rank() == 0:
            log.info('Training final model with best hyperparameters for evaluation.')

        # update the training hypers to the best values from the optimization
        for hyper, best_value in best_params.items():
            setattr(config_manager.hyperparams, hyper, best_value)

        # enable evaluation mode (train on train+valid and validate on test)
        config_manager.configuration.evaluation_mode = True

        # no need to validate
        config_manager.configuration.validation_frequency = 0

        # construct the final model
        trainer = TrainerFactory.get_trainer(config_manager)

        if torch.distributed.get_rank() == 0:
            log.info('!! Final training round on the full training dataset (train + valid) and evaluating on test.')
            log.info('--------------------------------------------------------------------------------------------')

        # fit model using training AND validation data. we also use the test
        # set for validation here.
        trainer.fit()

        # now we can evaluate the final performance of the best model
        if torch.distributed.get_rank() == 0:
            log.info('Evaluating final model on the test set.')

        if torch.distributed.get_rank() == 0:
            loss, metrics = trainer.test()
            log.info(f'Final loss: {loss:.4f}')

            # let's share the loss and metrics with other ranks
            # rank 0 is the source
            broadcast_objects = [loss, metrics]
        else:
            # other ranks receive
            broadcast_objects = [None, None]

        # now, broadcast the list from rank 0 to all the other ranks
        torch.distributed.broadcast_object_list(broadcast_objects, src=0)

        # now all the ranks have access to the metrics
        loss, metrics = broadcast_objects

        if metrics and torch.distributed.get_rank() == 0:
            log.info('Final metrics:')
            for key, value in metrics.items():
                log.info(f' - {key}: {value}.')

        log_final_epsilon(config_manager, trainer)

        return metrics

    @staticmethod
    def objective(
        config_manager: ConfigurationManager,
        optuna_config: dict,
        target_hypers: list,
        max_batch_size: int,
        process_group: torch.distributed.ProcessGroupGloo,
        trial: optuna.trial.Trial,
    ):
        # make the trial support distributed
        # https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py
        trial = optuna.integration.TorchDistributedTrial(trial, group=process_group)

        for target_hyper in target_hypers:
            log.info(f"target_hypers: {target_hypers}")
            log.info(f"optuna_config: {optuna_config}")
            if optuna_config[target_hyper]['type'] == 'float':
                hyper_value = trial.suggest_float(
                    target_hyper,
                    optuna_config[target_hyper]['min'],
                    optuna_config[target_hyper]['max'],
                    log=optuna_config[target_hyper].get('log_space', False),
                )

            if optuna_config[target_hyper]['type'] == 'int':
                max_value = optuna_config[target_hyper]['max']

                # Using -1 as max in batch size configuration signals full batch
                if target_hyper == 'batch_size' and optuna_config[target_hyper]['max'] == -1:
                    max_value = max_batch_size

                hyper_value = trial.suggest_int(
                    target_hyper,
                    optuna_config[target_hyper]['min'],
                    max_value,
                )

            if optuna_config[target_hyper]['type'] == 'categorical':
                hyper_value = trial.suggest_categorical(
                    target_hyper,
                    optuna_config[target_hyper]['options'],
                )

            # update the hyperparameter value in configuration
            setattr(config_manager.hyperparams, target_hyper, hyper_value)

        # train the modeol
        trainer = TrainerFactory.get_trainer(config_manager)

        # while tuning hypers, do not validate
        config_manager.configuration.validation_frequency = 0

        if torch.distributed.get_rank() == 0:
            log.info(f'Starting trial {trial.number}.')
            log.info(config_manager.hyperparams)

        trainer.fit()

        # optimization objective is the validation loss
        if torch.distributed.get_rank() == 0:
            loss, metrics = trainer.validate()

            log.info("Writing the loss and metrics of current trial into file.")
            save_hpo_metrics(
                config_manager,
                loss,
                metrics,
                trial_index=trial.number,
            )

            # let's share the loss and metrics with other ranks
            # rank 0 is the source
            broadcast_objects = [loss, metrics]
        else:
            # other ranks receive
            broadcast_objects = [None, None]

        # now, broadcast the list from rank 0 to all the other ranks
        torch.distributed.broadcast_object_list(broadcast_objects, src=0)

        # now all the ranks have access to the metrics
        loss, metrics = broadcast_objects

        # find the correct metric value to use as optimization objective
        target_metric = config_manager.configuration.optuna_target_metric
        if target_metric == 'loss':
            objective = loss
        elif target_metric in metrics:
            objective = metrics[target_metric]
        else:
            raise RuntimeError(f'Metric "{target_metric}" not found in metrics (' + ', '.join(metrics.keys()) + ')')

        # without these, this randomly leads to a nasty segmentation fault
        # on AMD, and a memory related CUDA error on Nvidia.
        del trainer, process_group, trial
        gc.collect()

        return objective

    @staticmethod
    def get_max_batch_size(config_manager: ConfigurationManager):
        datamodule = DataModuleFactory.get_datamodule(
            config_manager.configuration,
            config_manager.hyperparams,
        )
        num_classes = datamodule.get_num_classes()

        _, transforms = ModelFactory.get_model(
            config_manager.configuration,
            config_manager.hyperparams,
            num_classes,
        )

        datamodule.initialize(transforms)
        max_batch_size = datamodule.get_dataset_size()

        return max_batch_size
