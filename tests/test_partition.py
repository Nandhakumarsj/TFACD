import numpy as np
import pytest

from tfacd.data.partition import dirichlet_partition, iid_partition


def _skewed_labels(rng, majority=980, minority=5, num_classes=3):
    """A rare class (minority) sitting alongside a dominant one - the shape
    that let a class drop to 0 samples on some client at low alpha."""
    labels = np.concatenate(
        [
            np.zeros(majority, dtype=np.int64),
            np.ones(majority, dtype=np.int64),
            np.full(minority, 2, dtype=np.int64),
        ]
    )
    rng.shuffle(labels)
    return labels


def test_min_class_samples_disabled_by_default_preserves_prior_behavior():
    rng = np.random.default_rng(0)
    labels = _skewed_labels(rng)
    # Aggressive skew (low alpha) + a rare class is exactly the regime where a
    # class can vanish from a client - default min_class_samples=0 must not
    # raise over this, matching the pre-existing (unguarded) behavior.
    partitions = dirichlet_partition(labels, num_clients=5, alpha=0.1, seed=7, min_samples=1, max_attempts=50)
    assert len(partitions) == 5
    assert sum(len(p) for p in partitions) == len(labels)


def test_min_class_samples_guarantees_every_client_sees_every_class():
    rng = np.random.default_rng(0)
    labels = _skewed_labels(rng)
    partitions = dirichlet_partition(
        labels, num_clients=5, alpha=0.5, seed=42, min_samples=1, max_attempts=200, min_class_samples=1,
    )
    classes = np.unique(labels)
    for client_indices in partitions:
        client_labels = labels[client_indices]
        for cls in classes:
            assert (client_labels == cls).sum() >= 1, f"class {cls} missing from a client despite min_class_samples=1"


def test_min_class_samples_raises_when_unsatisfiable():
    rng = np.random.default_rng(0)
    labels = _skewed_labels(rng, minority=2)  # only 2 examples of the rare class, 5 clients
    with pytest.raises(RuntimeError, match="min_class_samples"):
        dirichlet_partition(
            labels, num_clients=5, alpha=0.5, seed=1, min_samples=1, max_attempts=10, min_class_samples=1,
        )


def test_iid_partition_unaffected():
    rng = np.random.default_rng(0)
    labels = _skewed_labels(rng)
    partitions = iid_partition(labels, num_clients=5, seed=0)
    assert sum(len(p) for p in partitions) == len(labels)
