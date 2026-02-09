import datasets
import logging
import torch
import torchvision
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from collections import Counter
from functools import partial
from PIL import Image

from .configurationmanager import Configuration, Hyperparameters
from .tabularprocess import preprocess_adult_census_income, preprocess_dutch_dataset, balance_classes

log = logging.getLogger(__name__)


class DataModule:
    def __init__(self,
        dataset_name: str = 'default-dataset',
        batch_size: int = 64,
        physical_batch_size: int = 64,
        num_workers: int = 4,
        seed: int = 0,
        privacy: bool = True,
        test_size: float = 0.1,
        validation_size: float = 0.1,
        split_seed: int = 42,
        evaluation_mode: bool = False,
        label_field: str = None,
        image_field: str = None,
        imbalance_factor: float = None,
        fairness_imbalance_class: float = None,
    ):

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.physical_batch_size = physical_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.privacy = privacy
        self.test_size = test_size
        self.val_size = validation_size
        self.split_seed = split_seed
        self.evaluation_mode = evaluation_mode
        self._image_field = image_field
        self._label_field = label_field
        self._imbalance_factor = imbalance_factor
        self._fairness_imbalance_class = fairness_imbalance_class

        self._dataloaders = {
            'train': None,
            'valid': None,
            'test': None,
        }

        # The _load_datasets method will fill this
        self.num_classes = None

        # Load datasets to memory
        if torch.distributed.get_rank() == 0:
            # First load the data only rank 0. This is because, the datasets
            # might need to be loaded over the network, and rank 0 can cache
            # them to disk.
            self._load_datasets()
            torch.distributed.barrier()
        else:
            # Other ranks wait here for rank 0 to do its job.
            torch.distributed.barrier()

            # Now other ranks can load them to memory directly from disk
            self._load_datasets()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialize(self, transforms: torchvision.transforms.transforms.Compose):
        self.transforms = transforms

        # Make sure all images are RGB
        self._add_rgb_transform()

        # Again, first do the initialization on rank 0, so it can cache everything
        # on disk without race conditions.
        # NB: There _might_ be some methods to speed this up using multiple GPUs.
        if torch.distributed.get_rank() == 0:
            self._initialize_datasets()
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            self._initialize_datasets()

        # we use batch size of -1 to signal full batch
        if self.batch_size == -1:
            self.batch_size = len(self.train_dataset)

        self._initialize_dataloaders()

    def get_num_classes(self):
        return self.num_classes

    def get_dataloader(self, name):
        return self._dataloaders.get(name)

    def get_dataset_size(self, which='train_dataset'):
        dataset = getattr(self, which)
        return len(dataset)

    def set_dataloader(self, name, dataloader):
        self._dataloaders[name] = dataloader

    def _default_collate_fn(batch):
        # default collate is a no-op
        return batch

    def _initialize_datasets(self):
        # Create datasets train/validation/test splits if they do not yet exists
        self._create_dataset_splits()

        # Imbalance before subsetting. If done in other order, we can get into
        # trouble with e.g. classes with zero examples
        if self._imbalance_factor and not self._fairness_imbalance_class:
            if torch.distributed.get_rank() == 0:
                log.info('Creating imbalanced train set..')

            self.train_dataset = self._get_imbalanced_subset(self.train_dataset)

            if torch.distributed.get_rank() == 0:
                log.info('Creating imbalanced validation set..')

            self.val_dataset = self._get_imbalanced_subset(self.val_dataset)

            if torch.distributed.get_rank() == 0:
                log.info('Creating imbalanced test set..')

            self.test_dataset = self._get_imbalanced_subset(self.test_dataset)

        # NOTE: we use full data for validation and test, but scale the metrics accordingly.
        if self._fairness_imbalance_class:
            if torch.distributed.get_rank() == 0:
                log.info("Creating fairness imbalanced train set..")
                self.train_dataset = self._get_fairness_imbalanced_subset(self.train_dataset)

            if torch.distributed.get_rank() == 0:
                log.info("Creating fairness imbalanced validation set..")
                self.val_dataset = self._get_fairness_imbalanced_subset(self.val_dataset)

            if torch.distributed.get_rank() == 0:
                log.info(f"We will not create fairness imbalanced test sets. Size of test set: {len(self.test_dataset)}")

    def _load_datasets(self):
        """Load the datasets to memory."""
        if torch.distributed.get_rank() == 0:
            log.info(f'Loading dataset "{self.dataset_name}" from Huggingface datasets.')

        dataset_splits = datasets.load_dataset(self.dataset_name)

        # Set dataset label fields based on the training split
        self._set_dataset_label_fields(dataset_splits)

        # Make sure the dataset label field is of type ClassLabel
        self._dataset_splits = self._enforce_label_field_type(dataset_splits)

        # Automatically determine the number of classes
        # NB: This can be done if the label is of type ClassLabel
        self.num_classes = dataset_splits['train'].features[self._label_field].num_classes

        if torch.distributed.get_rank() == 0:
            log.info(f'Determined the number of classes to be {self.num_classes}.')

    def _create_dataset_splits(self):
        # Check if there's a validation split available
        has_validation_split = 'validation' in self._dataset_splits
        has_test_split = 'test' in self._dataset_splits

        # We have all the splits, just use them as they are
        if has_validation_split and has_test_split:
            # Use separate validation set if it exists
            self.train_dataset = self._dataset_splits['train']
            self.val_dataset = self._dataset_splits['validation']
            self.test_dataset = self._dataset_splits['test']

        # No validation or test splist, create both
        if not has_validation_split and not has_test_split:
            # Split the training dataset into training and validation
            self.train_dataset, val_and_test_split = self._dataset_splits['train'].train_test_split(
                test_size=(self.test_size + self.val_size),
                seed=self.split_seed,
                shuffle=True,
                stratify_by_column=self._label_field,
            ).values()

            self.val_dataset, self.test_dataset = val_and_test_split.train_test_split(
                test_size=0.5,
                seed=self.split_seed,
                shuffle=True,
                stratify_by_column=self._label_field,
            ).values()

        # We have only test split, create validation split from train
        if not has_validation_split and has_test_split:
            # Split the training dataset into training and validation
            self.train_dataset, self.val_dataset = self._dataset_splits['train'].train_test_split(
                test_size=self.test_size,
                seed=self.split_seed,
                shuffle=True,
                stratify_by_column=self._label_field,
            ).values()

            self.test_dataset = self._dataset_splits['test']

        if has_validation_split and not has_test_split:
            raise ValueError('Splitting not implemented: Dataset has validation split but no test split.')

        if self.evaluation_mode:
            # Combine training and validation sets if we have a separate validation set
            self.train_dataset = datasets.concatenate_datasets([
                self.train_dataset,
                self.val_dataset,
            ])

            # In evaluation mode, we validate on the test dataset
            self.val_dataset = self.test_dataset

    def _enforce_label_field_type(self, dataset_splits):
        # Iterate through all dataset splits, and make the label field ClassLabel
        for key in dataset_splits.keys():
            dataset = dataset_splits[key]

            # If it already is a ClassLabel, HF dataset will throw an error, so check first
            if not isinstance(dataset.features[self._label_field], datasets.ClassLabel):
                dataset = dataset.class_encode_column(self._label_field)

            dataset_splits[key] = dataset

        return dataset_splits

    def _set_dataset_label_fields(self, dataset_splits):
        # extract the keys that contain the labels and images
        if torch.distributed.get_rank() == 0:
            log.info('Setting dataset fields.')

        self._set_image_field(dataset_splits['train'])
        self._set_label_field(dataset_splits['train'])

    def _set_image_field(self, dataset):
        if self._image_field is None:
            for feature_name, feature in dataset.features.items():
                if isinstance(feature, datasets.Image):
                    self._image_field = feature_name
                    break

            if self._image_field:
                if torch.distributed.get_rank() == 0:
                    log.info(f' - Determined image field: {self._image_field}')
            else:
                features = dataset.features.keys()
                raise ValueError('Could not determine image field for dataset.')

    def _set_label_field(self, dataset):
        if self._label_field is None:
            for feature_name, feature in dataset.features.items():
                if isinstance(feature, datasets.ClassLabel) or feature_name == 'label':
                    self._label_field = feature_name
                    break

            if self._label_field:
                if torch.distributed.get_rank() == 0:
                    log.info(f' - Determined label field: {self._label_field}')
            else:
                features = dataset.features.keys()
                raise ValueError('Could not determine label field for dataset. Available features: {features}')

    def _apply_transforms_to_datasets(self):
        return # no default transforms

    def _initialize_dataloaders(self):
        self._set_generators_and_seed_worker()
        self._create_dataloaders()

    def _set_generators_and_seed_worker(self):
        self.generator = torch.Generator()
        if self.seed:
            self.generator.manual_seed(self.seed)

        # each dataloader will get a different seed
        def seed_worker(worker_id):
            worker_seed = self.seed + worker_id
            torch.manual_seed(worker_seed)

        self.seed_worker = seed_worker if self.seed else None

    def _create_dataloaders(self):
        # We might need initialize a DataModule without a batch size,
        # at least in the case of figuring out the maximum batch size
        # from the dataset length.
        if not self.batch_size:
            if torch.distributed.get_rank() == 0:
                log.info('Batch size not yet initialized, skipping dataloader creation.')
            return

        self._set_samplers_and_batch_size()

        if self._collate_fn:
            # NB: The collate_fn needs to know the label and image fields,
            #     so let's overwrite it with a function that has those.
            #
            #     We also need to do the transformations in the collate function
            #     if we have not mapped the transformations to disk cache.
            collate_fn = partial(
                self._collate_fn,
                label_field=self._label_field,
                image_field=self._image_field,
                transforms=self.transforms,
            )
        else:
            collate_fn = self._default_collate_fn

        self._dataloaders['train'] = torch.utils.data.DataLoader(
            self.train_dataset.with_format('torch'),
            sampler=self.train_sampler,
            batch_size=self.local_batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=self.generator,
            worker_init_fn=self.seed_worker
        )

        self._dataloaders['valid'] = torch.utils.data.DataLoader(
            self.val_dataset.with_format('torch'),
            sampler=self.val_sampler,
            batch_size=self.physical_batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers
        )

        if self.test_dataset:
            self._dataloaders['test'] = torch.utils.data.DataLoader(
                self.test_dataset.with_format('torch'),
                sampler=self.test_sampler,
                batch_size=self.physical_batch_size,
                collate_fn=collate_fn,
                num_workers=self.num_workers
            )

    def _set_samplers_and_batch_size(self):
        # for the DP case, Opacus handles distributed for us. otherwise, we need
        # to use distributedsampler and divide the batch size by number of replicas
        if not self.privacy:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset.with_format('torch')
            )

            # For distributed without Opacus, we need to divide the batch size
            # by the world size.
            if self.batch_size:
                self.local_batch_size = self.batch_size // torch.distributed.get_world_size()
        else:
            # For the DP case, Opacus handles these for us
            self.train_sampler = None
            self.local_batch_size = self.batch_size

        # we will validate and test only on rank 0
        self.val_sampler, self.test_sampler = None, None

    def _get_fairness_imbalanced_subset(self, dataset):
        """
        Creates an imbalanced subset with one class having less examples than the others.
        """
        if not self._fairness_imbalance_class:
            raise ValueError(
                "Fairness imbalance class must be provided for creating an imbalanced dataset."
            )

        if self._imbalance_factor == 1.0:
            raise ValueError(
                "Imbalance factor must be less than 1.0 for creating a imbalanced dataset."
            )

        label_counts = Counter(dataset[self._label_field])
        label_counts = list(label_counts.values())
        num_classes = len(label_counts)

        img_num_per_cls = [
            (
                label_counts[i]
                if i != self._fairness_imbalance_class
                else int(label_counts[i] * self._imbalance_factor)
            )
            for i in range(num_classes)
        ]

        class_indices = {cls: [] for cls in range(num_classes)}
        for idx, sample in enumerate(dataset):
            class_indices[sample[self._label_field]].append(idx)

        sampled_indices = []
        for cls_idx, original_cls_idx in enumerate(class_indices):
            indices = class_indices[original_cls_idx]
            num_samples = img_num_per_cls[cls_idx]

            if num_samples > len(indices):
                log.warning(
                    f"Requested {num_samples} samples for class {original_cls_idx}, but only {len(indices)} available. Adjusting to {len(indices)}."
                )
                num_samples = len(indices)

            sampled_indices.extend(
                np.random.choice(indices, num_samples, replace=False)
            )

        # print(f"First 10 samples: {sampled_indices[:10]}")

        sampled_dataset = dataset.select(sampled_indices)

        if torch.distributed.get_rank() == 0:
            distribution = Counter(sampled_dataset[self._label_field])
            log.info(
                f"Created fairness imbalanced dataset (size: {len(sampled_dataset)}) with class distribution: {sorted(distribution.items())}"
            )

        return sampled_dataset


class ImageDataModule(DataModule):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def _apply_transforms_to_datasets(self):
        if torch.distributed.get_rank() == 0:
            log.info('Applying transformations to dataset.')

        def _apply_transforms(transforms, label_field, image_field, examples):
            log.info('.')
            examples[image_field] = [transforms(image) for image in examples[image_field]]
            return examples

        if self.transforms:
            transforms_func = partial(
                _apply_transforms,
                self.transforms,
                self._label_field,
                self._image_field,
            )

            if torch.distributed.get_rank() == 0:
                log.info(f' - Processing {len(self.train_dataset)} examples in the train dataset.')

            self.train_dataset = self.train_dataset.map(
                transforms_func,
                num_proc=self.num_workers,
                batched=True,
                load_from_cache_file=True,
            )

            if torch.distributed.get_rank() == 0:
                log.info(f' - Processing {len(self.val_dataset)} examples in the validation dataset.')

            self.val_dataset = self.val_dataset.map(
                transforms_func,
                num_proc=self.num_workers,
                batched=True,
                load_from_cache_file=True,
            )

            if self.test_dataset:
                if torch.distributed.get_rank() == 0:
                    log.info(f' - Processing {len(self.test_dataset)} examples in the test dataset.')

                self.test_dataset = self.test_dataset.map(
                    transforms_func,
                    num_proc=self.num_workers,
                    batched=True,
                    load_from_cache_file=True,
                )

    def _add_rgb_transform(self):
        def to_rgb_tensor(x):
            if isinstance(x, torch.Tensor):
                if len(x.shape) == 2:  # Grayscale tensor (H, W)
                    return x.unsqueeze(0)  # Add a channel dimension to make it (1, H, W)
                elif len(x.shape) == 3 and x.shape[0] == 1:
                    return x  # Already grayscale with (1, H, W) shape
                    # return x.repeat(3, 1, 1)
                elif len(x.shape) == 3 and x.shape[0] == 3:
                    return x.mean(dim=0, keepdim=True)  # Convert RGB to grayscale with (1, H, W) shape
                    # return x
                else:
                    raise ValueError('Input tensor is not a valid image tensor.')
            return x

        toRGB = torchvision.transforms.Lambda(to_rgb_tensor)

        # Update the transform pipeline
        new_transforms = [toRGB] + self.transforms.transforms
        self.transforms = torchvision.transforms.Compose(new_transforms)

    def _replace_to_tensor_with_to_float(self):
        # We need to convert the tensor to float and normalize to [0, 1]
        to_float = torchvision.transforms.Lambda(lambda x: x.float() / 255.0)

        # Filter out ToTensor and replace it with the new transformation
        new_transforms = []
        for t in self.transforms.transforms:
            if isinstance(t, torchvision.transforms.ToTensor):
                new_transforms.append(to_float)
            else:
                new_transforms.append(t)

        self.transforms.transforms = new_transforms

    @staticmethod
    def _collate_fn(batch, label_field=None, image_field=None, transforms=None):
        B = len(batch)

        # Apply transformation to the first image to determine the size after transformation
        if transforms:
            first_image = transforms(batch[0][image_field])
        else:
            first_image = batch[0][image_field]

        # Now that we know the transformed image size, we can initialize the `images` tensor
        C, H, W = first_image.shape
        images = torch.empty((B, C, H, W))
        labels = torch.empty(B, dtype=torch.long)

        # Now we are ready to process the batch
        for i in range(B):
            if transforms:
                images[i] = transforms(batch[i][image_field])
            else:
                images[i] = batch[i][image_field]

            labels[i] = batch[i][label_field]

        return images, labels

    @staticmethod
    def _collate_fn_with_cached_features(batch, label_field=None, image_field=None):
        features = torch.stack(
            [item['features'] for item in batch]
        )

        labels = torch.tensor(
            [item[label_field] for item in batch]
        )

        return features, labels

class TabularDataModule(DataModule):
    def __init__(
        self,
        target_column: str,
        protected_column: str,
        val_rate: float = 0.3,
        **kwargs
    ):
        self.target_column = target_column
        self.protected_column = protected_column
        self.val_rate = val_rate
        super().__init__(**kwargs)
        self.num_classes = 2

    def _add_rgb_transform(self):
        pass

    def _replace_to_tensor_with_to_float(self):
        pass

    def _create_dataset_splits(self):
        pass

    def _load_datasets(self):
        """Load and preprocess datasets."""
        if torch.distributed.get_rank() == 0:
            log.info(f'Loading tabular dataset "{self.dataset_name}".')

        # Load and preprocess the dataset
        if self.dataset_name == "adult":
            df = pd.read_csv("./data/adult/adult.csv", sep=",")
            df = preprocess_adult_census_income(df)
        elif self.dataset_name == "dutch":
            df = pd.read_csv("./data/dutch/dutch.csv", sep=",")
            df = preprocess_dutch_dataset(df)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        # balance classes
        train_val, test = balance_classes(df, base_column=self.protected_column)
        index_protected_column = df.columns.get_loc(self.protected_column)

        # split the features and target
        train_val_X = train_val.drop(self.target_column, axis=1)
        train_val_y = train_val[self.target_column]

        test_X = test.drop(self.target_column, axis=1)
        test_y = test[self.target_column]

        X_train, X_val, y_train, y_val = train_test_split(
            train_val_X.values,
            train_val_y.values,
            test_size=self.val_rate,
            random_state=self.seed,
        )

        self.train_dataset = list(
            zip(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            )
        )
        self.val_dataset = list(
            zip(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            )
        )
        self.test_dataset = list(
            zip(
                torch.tensor(test_X.values, dtype=torch.float32),
                torch.tensor(test_y.values, dtype=torch.float32),
            )
        )

    def _initialize_dataloaders(self):
        """Create DataLoader instances for train and validation datasets."""
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        self._dataloaders = {
            'train': self.train_dataloader,
            'valid': self.val_dataloader,
            'test': self.test_dataloader,
        }

    def get_dataloader(self, name):
        return self._dataloaders.get(name)

    def get_dataset(self, is_train=True):
        """Return the dataset for training or validation."""
        return self.train_dataset if is_train else self.val_dataset

    def initialize(self, transforms: torchvision.transforms.transforms.Compose):
        self.transforms = transforms

        if self.transforms:
            self._replace_to_tensor_with_to_float()

        if torch.distributed.get_rank() == 0:
            self._initialize_datasets()
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            self._initialize_datasets()

        if self.batch_size == -1:
            self.batch_size = len(self.train_dataset)

        self._initialize_dataloaders()


class DataModuleFactory:
    @staticmethod
    def get_datamodule(
        configuration: Configuration,
        hyperparams: Hyperparameters,
    ) -> DataModule:

        if configuration.dataset_name not in ["adult", "dutch"]:
            datamodule = ImageDataModule(
                dataset_name=configuration.dataset_name,
                num_workers=configuration.num_workers,
                physical_batch_size=configuration.physical_batch_size,
                validation_size=configuration.validation_size,
                test_size=configuration.test_size,
                seed=configuration.seed,
                batch_size=hyperparams.batch_size,
                privacy=configuration.privacy,
                evaluation_mode=configuration.evaluation_mode,
                label_field=configuration.dataset_label_field,
                imbalance_factor=configuration.imbalance_factor,
                fairness_imbalance_class=configuration.fairness_imbalance_class,
            )
        else:
            datamodule = TabularDataModule(
                dataset_name=configuration.dataset_name,
                target_column=configuration.dataset_label_field,
                protected_column=configuration.protected_feature,
                num_workers=configuration.num_workers,
                physical_batch_size=configuration.physical_batch_size,
                validation_size=configuration.validation_size,
                test_size=configuration.test_size,
                seed=configuration.seed,
                batch_size=hyperparams.batch_size,
                privacy=configuration.privacy,
                evaluation_mode=configuration.evaluation_mode,
                label_field=configuration.dataset_label_field,
                imbalance_factor=configuration.imbalance_factor,
                fairness_imbalance_class=configuration.fairness_imbalance_class,
            )

        return datamodule
