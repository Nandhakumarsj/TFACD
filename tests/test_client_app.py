import numpy as np
import torch
from torch.utils.data import DataLoader

from tfacd.data.dataset import SequenceDataset
from tfacd.federated.client_app import _weighted_criterion


def _loader_with_labels(labels):
    y = np.array(labels, dtype=np.int64)
    x = np.zeros((len(y), 1, 3), dtype=np.float32)
    return DataLoader(SequenceDataset(x, y), batch_size=4)


def test_weighted_criterion_applies_local_class_weights_by_default():
    # Heavily imbalanced local partition: class 0 dominates, class 2 is rare.
    loader = _loader_with_labels([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2])
    config = {"training": {}}  # class_weighting defaults True, matching training/centralized.py's cfg.get(..., True)

    criterion = _weighted_criterion(config, loader, torch.device("cpu"), num_classes=3)

    assert criterion.weight is not None
    assert criterion.weight.shape == (3,)
    # The rare class (2) must get a strictly larger weight than the dominant class (0).
    assert criterion.weight[2] > criterion.weight[0]


def test_weighted_criterion_respects_class_weighting_false():
    loader = _loader_with_labels([0, 0, 0, 1, 2])
    config = {"training": {"class_weighting": False}}

    criterion = _weighted_criterion(config, loader, torch.device("cpu"), num_classes=3)

    assert criterion.weight is None


def test_weighted_criterion_missing_config_key_defaults_to_weighted():
    loader = _loader_with_labels([0, 0, 0, 1, 2])
    config = {"training": {}}

    criterion = _weighted_criterion(config, loader, torch.device("cpu"), num_classes=3)

    assert criterion.weight is not None
