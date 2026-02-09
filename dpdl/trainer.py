import logging
import math
import opacus
import torch
import torchmetrics

from opacus.privacy_engine import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from .models.model_factory import ModelFactory
from .callbacks.callback_factory import CallbackHandler, CallbackFactory
from .configurationmanager import ConfigurationManager, Configuration, Hyperparameters
from .datamodules import DataModule, DataModuleFactory
from .optimizers import OptimizerFactory
from .utils import seed_everything

log = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,

        # essentials
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        datamodule: DataModule,

        # generic params
        epochs: int = 10,
        total_steps: int = None,
        validation_frequency: int = 1,
        seed: int = 0,
        physical_batch_size: int = 40,
        callback_handler: CallbackHandler = None,
    ):

        self.model = model
        self.optimizer = optimizer
        self.datamodule = datamodule
        self.epochs = epochs
        self.total_steps = total_steps
        self.validation_frequency = validation_frequency
        self.seed = seed
        self.physical_batch_size = physical_batch_size

        if not callback_handler:
            self.callback_handler = CallbackHandler()
        else:
            self.callback_handler = callback_handler

        if self.epochs and self.total_steps:
            raise ValueError('You should provide either "epochs" or "total_steps", not both.')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup()

    def setup(self):
        self.model = self.model.to(self.device)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model)

    def fit(self):
        self.callback_handler.call('on_train_start', self)

        if self.total_steps:
            self._fit_total_steps()
        else:
            self._fit_epochs()

        self.callback_handler.call('on_train_end', self)

    def _fit_epochs(self):
        for epoch in range(self.epochs):
            self.fit_one_epoch(epoch)

            if self.validation_frequency and epoch % self.validation_frequency == 0:
                if torch.distributed.get_rank() == 0:
                    self.validate(epoch)

                # other ranks will wait for validation
                torch.distributed.barrier()

    def _fit_total_steps(self):
        step = 0
        virtual_epoch = 0
        steps_per_epoch = self._calculate_steps_per_epoch()

        # start the first virtual epoch
        self._handle_virtual_epoch_start(virtual_epoch)

        while step < self.total_steps:
            for batch_idx, batch in enumerate(self.datamodule.get_dataloader('train')):
                if step >= self.total_steps:
                    break

                self.fit_one_batch(batch_idx, batch)
                step += 1

                if step % steps_per_epoch == 0:
                    self._handle_virtual_epoch_end(virtual_epoch)
                    virtual_epoch += 1

                    if self.validation_frequency and virtual_epoch % self.validation_frequency == 0:
                        if torch.distributed.get_rank() == 0:
                            self.validate(virtual_epoch)

                        # other ranks will wait for validation
                        torch.distributed.barrier()

                    # are we finished?
                    if step >= self.total_steps:
                        break

                    # start the next virtual epoch
                    self._handle_virtual_epoch_start(virtual_epoch)

        assert step == self.total_steps, f'Mismatch in total steps count: Expected {self.total_steps} total steps, but stepped {step} times!'

    def _handle_virtual_epoch_start(self, epoch):
        self.callback_handler.call('on_train_epoch_start', self, epoch)

    def _handle_virtual_epoch_end(self, epoch):
        # compute the epoch metrics
        metrics = self._unwrap_model().train_metrics.compute()

        self._unwrap_model().train_metrics.reset()
        if self.use_fairness_metrics:
            extra_metrics = self._unwrap_model().extra_train_metrics.compute()
            self._unwrap_model().extra_train_metrics.reset()
            metrics = {**metrics, **extra_metrics}

        self.callback_handler.call('on_train_epoch_end', self, epoch, metrics)

    def fit_one_epoch(self, epoch):
        self.model.train()
        self.callback_handler.call('on_train_epoch_start', self, epoch)

        for batch_idx, batch in enumerate(self.datamodule.get_dataloader('train')):
            self.callback_handler.call('on_train_batch_start', self, batch_idx, batch)
            self.fit_one_batch(batch_idx, batch)
            self.callback_handler.call('on_train_batch_end', self, batch_idx, batch, logical_batch_loss)

        # compute the epoch metrics
        metrics = self._unwrap_model().train_metrics.compute()
        self._unwrap_model().train_metrics.reset()

        self.callback_handler.call('on_train_epoch_end', self, epoch, metrics)

    def fit_one_batch(self, batch_idx, batch):
        X, y = batch
        X = X.cuda(non_blocking=True) if self.device == 'cuda' else X
        y = y.cuda(non_blocking=True) if self.device == 'cuda' else y

        # gradient accumulation. split the batch to sub batches that fit in the GPU memory.
        # then process the sub batches one at a time and call backward.
        # when all the sub batches have been processed we can finally step the optimizer.
        X_split = X.split(self.physical_batch_size, dim=0)
        y_split = y.split(self.physical_batch_size, dim=0)

        # zero the grads as usually before doing anything
        self.optimizer.zero_grad()

        logical_batch_loss = 0
        # process the sub batches one at a time
        N = len(X_split)

        for i in range(N):
            # notify the callbacks of a physical batch start
            X_splitted = X_split[i]
            y_splitted = y_split[i]
            physical_batch = (X_splitted, y_splitted)
            if self.use_fairness_metrics:
                protected_feature = X_splitted[:, -1]
                protected_feature = protected_feature.cuda(non_blocking=True) if self.device == 'cuda' else protected_feature

            self.callback_handler.call('on_train_physical_batch_start', self, i, physical_batch)

            logits = self.model(X_splitted)
            loss = self._unwrap_model().criterion(logits, y_splitted) / N # NB: normalize loss
            loss.backward()

            # keep track of the batch loss
            logical_batch_loss += loss.item() * N # NB: Unnormalize to track logical batch loss

            # update the metrics if there are any
            preds = torch.argmax(logits, dim=1)
            self._unwrap_model().train_metrics.update(preds, y_split[i])
            if self.use_fairness_metrics:
                self._unwrap_model().extra_train_metrics.update(logits, y, protected_feature)

            # notify the callbacks of a physical batch end
            self.callback_handler.call('on_train_physical_batch_end', self, i, physical_batch, loss.item())

        # after accumulating the gradients for all the sub batches we can finally update weights.
        self.optimizer.step()

        return logical_batch_loss

    def validate(self, epoch=None):
        return self._evaluate('validation', epoch)

    def test(self):
        return self._evaluate('test')

    def get_dataloader(self, name):
        return self.datamodule.get_dataloader(name)

    def get_datamodule(self):
        return self.datamodule

    def _evaluate(self, mode, epoch=None):
        self.callback_handler.call(f'on_{mode}_epoch_start', self, epoch)
        self.model.eval()
        torch.set_grad_enabled(False)

        # record the loss separately, as we need to return it when
        # performing hyperparameter optimization
        evaluation_loss = 0

        dataloader_name = 'valid' if mode == 'validation' else 'test'
        dataloader = self.datamodule.get_dataloader(dataloader_name)

        for batch_idx, batch in enumerate(dataloader):
            loss = self._evaluate_one_batch(mode, batch_idx, batch)
            evaluation_loss += loss

        evaluation_loss /= len(dataloader)

        metrics = self._unwrap_model().valid_metrics.compute()
        if self.use_fairness_metrics:
            extra_metrics = self._unwrap_model().extra_valid_metrics.compute()
            self._unwrap_model().extra_valid_metrics.reset()
            metrics = {**metrics, **extra_metrics}

        self._unwrap_model().valid_metrics.reset()

        torch.set_grad_enabled(True)
        self.model.train()

        self.callback_handler.call(f'on_{mode}_epoch_end', self, epoch, metrics)

        return evaluation_loss, metrics

    def _evaluate_one_batch(self, mode, batch_idx, batch):
        self.callback_handler.call(f'on_{mode}_batch_start', self, batch_idx, batch)

        X, y = batch
        X = X.cuda(non_blocking=True) if self.device == 'cuda' else X
        y = y.cuda(non_blocking=True) if self.device == 'cuda' else y

        if self.use_fairness_metrics:
            protected_feature = X[:, -1]
            protected_feature = protected_feature.cuda(non_blocking=True) if self.device == 'cuda' else protected_feature

        logits = self.model(X)
        loss = self._unwrap_model().criterion(logits, y)
        loss = loss.item()

        preds = torch.argmax(logits, dim=1)
        self._unwrap_model().valid_metrics.update(preds, y)
        if self.use_fairness_metrics:
            self._unwrap_model().extra_valid_metrics.update(logits, y, protected_feature)

        self.callback_handler.call(f'on_{mode}_batch_end', self, batch_idx, batch, loss)
        return loss

    def _unwrap_model(self):
        # the model is wrapped inside torch distributed,
        # here we just return the vanilla model
        return self.model.module

    def _calculate_steps_per_epoch(self):
        data_size = len(self.datamodule.get_dataloader('train').dataset)
        batch_size = self.datamodule.batch_size

        return math.ceil(data_size / batch_size)

    def save_model(self, fpath):
        self.model.save_model(fpath)

class DifferentiallyPrivateTrainer(Trainer):
    def __init__(
        self,
        *,
        # privacy params
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        clipping_mode: str = 'flat',
        accountant: str = 'prv',
        poisson_sampling: bool = True,
        normalize_clipping: bool = False,
        secure_mode: bool = False,
        target_epsilon: float = 0,
        target_delta: float = 0,
        seed: int = 0,
        record_grad_and_noise: bool = False,
        optim_args: dict = None,
        use_fairness_metrics: bool = False,
        **kwargs,
    ):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.clipping_mode = clipping_mode
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.seed = seed
        self.poisson_sampling = poisson_sampling
        self.normalize_clipping = normalize_clipping
        self.optim_args = optim_args
        self.use_fairness_metrics = use_fairness_metrics

        # setup opacus privacy engine
        privacy_engine_args = {
            'accountant': accountant,
            'secure_mode': secure_mode,
        }

        self.privacy_engine = PrivacyEngine(**privacy_engine_args)

        super().__init__(seed=seed, **kwargs)

    def _has_target_privacy_params(self):
        if self.target_epsilon == -1:
            return False

        if not any([self.target_epsilon, self.target_delta]):
            return False

        if self.target_epsilon and not self.target_delta:
            raise RuntimeError('Parameter "target_delta" present, but "target_epsilon" is missing.')

        if self.target_delta and not self.target_epsilon:
            raise RuntimeError('Parameter "target_delta" present, but "target_epsilon" is missing.')

        if self.target_epsilon and self.noise_multiplier:
            raise RuntimeError('Parameter "noise_multiplier" can not be used when target epsilon is given.')

        return True

    def setup(self):
        noise_generator = torch.Generator(device=torch.cuda.current_device() if self.device == 'cuda' else 'cpu')
        if self.seed:
            noise_generator.manual_seed(self.seed)

        self.model = self.model.to(self.device)

        # let's be distributed by default and wrap the model for Opacus DDP.
        # DifferentiallyPrivateDistributedDataParallel is actually a no-op in Opacus, but
        # let's wrap anyway in case of future api changes. https://opacus.ai/tutorials/ddp_tutorial
        model = opacus.distributed.DifferentiallyPrivateDistributedDataParallel(self.model)

        optimizer = self.optimizer
        train_dataloader = self.datamodule.get_dataloader('train')

        # setup differential privacy for the model, optimize, and dataloader
        if self._has_target_privacy_params():
            dp_model, dp_optimizer, dp_dataloader = self.privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                max_grad_norm=self.max_grad_norm,
                clipping=self.clipping_mode,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                epochs=self.epochs,
                noise_generator=noise_generator,
                poisson_sampling=self.poisson_sampling,
                normalize_clipping=self.normalize_clipping,
                total_steps=self.total_steps,
                optim_args=self.optim_args,
            )
        else:
            if self.target_epsilon == -1:
                self.noise_multiplier = 0

            dp_model, dp_optimizer, dp_dataloader = self.privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                clipping=self.clipping_mode,
                noise_generator=noise_generator,
                poisson_sampling=self.poisson_sampling,
                normalize_clipping=self.normalize_clipping,
                total_steps=self.total_steps,
                optim_args=self.optim_args,
            )

        # now we can start using the DP'ifyed stuff
        self.model = dp_model
        self.datamodule.set_dataloader('train', dp_dataloader)
        self.optimizer = dp_optimizer

    def get_epsilon(self):
        return self.privacy_engine.get_epsilon(self.target_delta)

    def _unwrap_model(self):
        # the model is wrapped inside Opacus, and Opacus distributed.
        # let's unwrap the vanilla model and return it
        return self.model._module.module

    def _fit_total_steps(self):
        # here we'll keep track of our approximate epochs
        virtual_epoch = 0

        # number of total steps taken
        step = 0

        # number of logical batches in an approximate epoch
        n_logical_batches = 0

        # track the logical batch loss here
        logical_batch_loss = 0

        # track the number of physical batches in a logical batch
        n_physical_batch_in_logical = 0

        # flag to indicate the beginning of a new logical batch
        logical_batch_begin = True

        # flag to indicate that a logical batch has been completed (set via the optimizer check)
        logical_batch_completed = False

        # to calculate the start/end of an epoch, we need the number
        # of steps in an epoch.
        steps_per_epoch = self._calculate_steps_per_epoch()

        # At the very start, call on_train_batch_start for the first logical batch.
        if logical_batch_begin:
            self.callback_handler.call('on_train_batch_start', self, n_logical_batches, None)
            logical_batch_begin = False

        # if 'total_steps' is set then Opacus will do the stepping for us, or
        # more precisely: the dataloader will have exactly 'total_steps' batches.
        # Here, we will spend approximately an epoch worth of those.
        with BatchMemoryManager(
            data_loader=self.datamodule.get_dataloader('train'),
            max_physical_batch_size=self.physical_batch_size,
            optimizer=self.optimizer,
        ) as virtual_dataloader:
            for batch_idx, batch in enumerate(virtual_dataloader):
                # first batch, we can start first epoch
                if batch_idx == 0:
                    self._handle_virtual_epoch_start(virtual_epoch)

                # now, let's check if we are going to reach the end of logical batch.
                # the optimizer will not skip next gradient update if we are not at
                # the end of the logical batch. there's currently pretty much no other
                # way to do it than this, becaues we don't know the size of the logical
                # batch that was sampled.
                if not self.optimizer._check_skip_next_step(False):
                    step += 1
                    logical_batch_completed = True
                else:
                    logical_batch_completed = False

                # notify the callbacks of a physical batch start
                self.callback_handler.call('on_train_physical_batch_start', self, batch_idx, batch)

                # let's fit this physical batch
                batch_loss = self.fit_one_batch(batch_idx, batch)

                # notify the callbacks of a physical batch end
                self.callback_handler.call('on_train_physical_batch_end', self, batch_idx, batch, batch_loss)

                logical_batch_loss += batch_loss
                n_physical_batch_in_logical += 1

                # if the logical batch is complete, notify batch end and reset counters
                if logical_batch_completed:
                    self.callback_handler.call(
                        'on_train_batch_end',
                        self,
                        n_logical_batches,
                        None,
                        logical_batch_loss / n_physical_batch_in_logical,  # mean of physical batch losses
                    )
                    n_logical_batches += 1
                    logical_batch_loss = 0
                    n_physical_batch_in_logical = 0

                    # the next iteration starts a new logical batch
                    logical_batch_begin = True

                # At the beginning of a new logical batch, call on_train_batch_start.
                if logical_batch_begin:
                    self.callback_handler.call('on_train_batch_start', self, n_logical_batches, None)
                    logical_batch_begin = False

                # and next we check for epoch end
                if (logical_batch_completed and step % steps_per_epoch == 0) or step == self.total_steps:
                    self._handle_virtual_epoch_end(virtual_epoch)

                    if self.validation_frequency and virtual_epoch % self.validation_frequency == 0:
                        # validate only on rank 0. no need to do distributed here,
                        # the computation is not heavy because we don't need gradients.
                        if torch.distributed.get_rank() == 0:
                            self.validate(virtual_epoch)

                        # other ranks will wait for validation
                        torch.distributed.barrier()

                    if step < self.total_steps:
                        virtual_epoch += 1
                        self._handle_virtual_epoch_start(virtual_epoch)
                        # Start a new logical batch for the new epoch.
                        self.callback_handler.call('on_train_batch_start', self, n_logical_batches, None)
                        logical_batch_begin = False

                # Reset the logical batch completion flag for the next iteration.
                logical_batch_completed = False

        if step != self.total_steps:
            log.warn(f'Was going to step for {self.total_steps}, but stepped only {step} steps.')

    def fit_one_batch(self, batch_idx, batch):
        self.optimizer.zero_grad()

        X, y = batch
        X = X.cuda(non_blocking=True) if self.device == 'cuda' else X
        y = y.cuda(non_blocking=True) if self.device == 'cuda' else y

        if self.use_fairness_metrics:
            protected_feature = X[:, -1]
            protected_feature = protected_feature.cuda(non_blocking=True) if self.device == 'cuda' else protected_feature

        logits = self.model(X)
        loss = self._unwrap_model().criterion(logits, y)

        loss.backward()

        self.optimizer.step()

        loss = loss.item()

        # update metrics if there are any
        preds = torch.argmax(logits, dim=1)

        self._unwrap_model().train_metrics.update(preds, y)
        if self.use_fairness_metrics:
            self._unwrap_model().extra_train_metrics.update(logits, y, protected_feature)

        return loss

    def fit_one_epoch(self, epoch):
        self.model.train()
        self.callback_handler.call('on_train_epoch_start', self, epoch)

        with BatchMemoryManager(
            data_loader=self.datamodule.get_dataloader('train'),
            max_physical_batch_size=self.physical_batch_size,
            optimizer=self.optimizer,
        ) as virtual_dataloader:
            # the virtual data loader created by BatchMemoryManager enables us to use larger
            # logical batch sizes that fit in a GPU.
            for batch_idx, batch in enumerate(virtual_dataloader):
                self.callback_handler.call('on_train_batch_start', self, batch_idx, batch)
                batch_loss = self.fit_one_batch(batch_idx, batch)
                self.callback_handler.call('on_train_batch_end', self, batch_idx, batch, batch_loss)

        # compute and reset the epoch metrics
        metrics = self._unwrap_model().train_metrics.compute()
        if self.use_fairness_metrics:
            extra_metrics = self._unwrap_model().extra_train_metrics.compute()
            self._unwrap_model().extra_train_metrics.reset()
            metrics = {**metrics, **extra_metrics}
        self._unwrap_model().train_metrics.reset()

        self.callback_handler.call('on_train_epoch_end', self, epoch, metrics)

    def save_model(self, fpath):
        self.model.module.save_model(fpath)

class TrainerFactory:
    @staticmethod
    def get_trainer(config_manager: ConfigurationManager) -> Trainer:

        if seed := config_manager.configuration.privacy:
            seed_everything(seed)

        # are we differentially private?
        if config_manager.configuration.privacy:
            return TrainerFactory._get_differentially_private_trainer(config_manager.configuration, config_manager.hyperparams)

    @staticmethod
    def _get_differentially_private_trainer(configuration: Configuration, hyperparams: Hyperparameters) -> Trainer:
        # Target delta calculation: A common heuristic is to use 1/N', with N'
        # being the size of the dataset rounded up to the nearest power of 10.
        # To avoid too large values of delta, let's pick a somewhat sensible
        # minimum of 1e-5.
        def _round_up_to_nearest_power_of_10(n):
            return 10 ** math.ceil(math.log10(n))

        def _calculate_target_delta(N):
            N_prime = _round_up_to_nearest_power_of_10(N)
            return min(1e-5, 1 / N_prime)

        def _get_target_privacy_params(hyperparams):
            # are we given a target epsilon?
            if hyperparams.target_epsilon is not None:
                N = len(datamodule.get_dataloader('train').dataset)
                target_delta = _calculate_target_delta(N)
                target_epsilon = hyperparams.target_epsilon
            else:
                target_delta = None
                target_epsilon = None

            return target_delta, target_epsilon

        def _get_epochs_and_steps(configuration, hyperparams, datamodule):
            # Differentially Private Learning with Adaptive Clipping
            # Andrew, et al.
            # https://arxiv.org/abs/1905.03871

            noise_multiplier_andrew = (
                datamodule.batch_size / hyperparams.count_noise_denom
            )
            optimizers_args = {
                'target_unclipped_quantile': hyperparams.target_quantile,
                'count_threshold': hyperparams.count_threshold,
                'clipbound_learning_rate': hyperparams.clip_bound_lr,
                'max_clipbound': float('inf'),
                'min_clipbound': hyperparams.clip_bound_lower_bound,
                'unclipped_num_std': noise_multiplier_andrew,
                'init_clipbound': hyperparams.clip_bound_init,
            }
            return optimizers_args

        # First initialize the DataModule, it will know about the number of classes
        datamodule = DataModuleFactory.get_datamodule(configuration, hyperparams)
        num_classes = datamodule.get_num_classes()

        # Now, setup data, model, and optimizer
        model, transforms = ModelFactory.get_model(configuration, hyperparams, num_classes)
        optimizer = OptimizerFactory.get_optimizer(configuration, hyperparams, model)

        # The datamodule needs to be aware of the transformations, now we can initialize it
        datamodule.initialize(transforms)

        callback_handler = CallbackHandler(
            CallbackFactory.get_callbacks(configuration, hyperparams)
        )

        target_delta, target_epsilon = _get_target_privacy_params(hyperparams)
        epochs, total_steps = TrainerFactory._get_epochs_and_steps(configuration, hyperparams, datamodule)

        optimizers_args = _get_epochs_and_steps(configuration, hyperparams, datamodule)

        # instantiate a differentialy private trained
        trainer = DifferentiallyPrivateTrainer(
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            # hypers
            epochs=epochs,
            total_steps=total_steps,
            noise_multiplier=hyperparams.noise_multiplier,
            max_grad_norm=hyperparams.max_grad_norm,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            poisson_sampling=configuration.poisson_sampling,
            normalize_clipping=configuration.normalize_clipping,
            # config
            accountant=configuration.accountant,
            secure_mode=configuration.secure_mode,
            clipping_mode=configuration.clipping_mode,
            physical_batch_size=configuration.physical_batch_size,
            seed=configuration.seed,
            callback_handler=callback_handler,
            validation_frequency=configuration.validation_frequency,
            optim_args=optimizers_args,
            use_fairness_metrics=True if configuration.protected_feature else False,
        )

        log.info(f'The arguments of the optimizer: {optimizers_args}')

        return trainer

    @staticmethod
    def _get_epochs_and_steps(
        configuration: Configuration,
        hyperparams: Hyperparameters,
        datamodule: DataModule,
    ):
        # use steps instead of epochs?
        if configuration.use_steps and hyperparams.epochs:
            B = datamodule.batch_size

            dataloader = datamodule.get_dataloader('train')
            N = len(dataloader.dataset)

            total_steps = math.ceil((N*hyperparams.epochs) / B)
            epochs = None
        # is the number of steps limited?
        elif configuration.use_steps and hyperparams.total_steps:
            total_steps = hyperparams.total_steps
            epochs = None
        # normal training using epochs
        else:
            total_steps = None
            epochs = hyperparams.epochs

        return epochs, total_steps
