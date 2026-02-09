# Code for reproducing the results in the paper "Mitigating Disparate Impact of Differentially Private Learning through Bounded Adaptive Clipping"

Our code here is a Lite version of another private project.

## Environment and Modified Libraries

- `Opacus`, the adaptive optimizers in opacus are modified

You can install Our Opacus by running:

`cd ./opacus && pip install -e .`

The full list of dependencies is in `requirements.txt`.

## Enterpoint

1. You can run the grid search with `sh 0_image.sh` or `sh 1_tabular.sh` for image and text datasets respectively.
2. You can run the DPHPO with Random Sampler with `sh 2_optim.sh`.

## Note

To keep the consistency with the Opacus and previous paper, the `max_grad_norm` is used as the clipping bound for the constant clipping, while the `clip_bound_lower_bound` is the lower bound for adaptive clipping.

## The Output

- `stdout.txt` and `stderr.txt`: log files.
- `test_metrics`: the test metrics.
- `configuration.*`: the configurations used in the run.
- `hyperparameters.*`: the hyperparameters used in the run.

### Image Datasets

The `MulticlassAccuracy` is the MacroAccuracy, while the Worst metric can be post-processing with `MulticlassAccuracyPerClass`.

### Tabular Datasets

The log file should contains lots of metrics. In the Tabular datasets, the `AccuracyOfProtectedGroups` should be the target metric.
