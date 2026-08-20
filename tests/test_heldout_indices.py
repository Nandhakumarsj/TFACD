import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tfacd.data.preprocess import heldout_indices

CLASSES = ["Normal", "DDoS_TCP", "MITM", "Port_Scanning"]


def _make_csv(tmp_path, n=200, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(CLASSES, size=n, p=[0.6, 0.2, 0.1, 0.1])
    frame = pd.DataFrame({"some_feature": rng.normal(size=n), "Attack_label": (labels != "Normal").astype(int), "Attack_type": labels})
    path = tmp_path / "synthetic.csv"
    frame.to_csv(path, index=False)
    return path, labels


def _expected_split(labels, seed=42, test_size=0.2, val_size=0.2):
    y = LabelEncoder().fit_transform(labels.astype(str))
    idx = np.arange(len(y))
    idx_trainval, idx_test, y_trainval, _ = train_test_split(idx, y, test_size=test_size, random_state=seed, stratify=y)
    return idx_test


def make_config(csv_path, output_dir, test_size=0.2, val_size=0.2, seed=42):
    return {
        "seed": seed,
        "data": {
            "raw_csv": str(csv_path), "output_dir": str(output_dir), "label_column": "auto",
            "test_size": test_size, "validation_size": val_size, "max_rows": None,
        },
    }


def test_reproduces_expected_split(tmp_path):
    csv_path, labels = _make_csv(tmp_path)
    config = make_config(csv_path, tmp_path / "out")
    expected = _expected_split(labels)
    actual = heldout_indices(config)
    assert np.array_equal(actual, expected)


def test_passes_without_prepared_npz_present(tmp_path):
    csv_path, _ = _make_csv(tmp_path)
    config = make_config(csv_path, tmp_path / "out_missing")
    # No prepared.npz written under out_missing - should not raise.
    result = heldout_indices(config)
    assert len(result) > 0


def test_asserts_against_a_matching_prepared_npz(tmp_path):
    csv_path, labels = _make_csv(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    expected = _expected_split(labels)
    y = LabelEncoder().fit_transform(labels.astype(str))
    np.savez_compressed(out / "prepared.npz", y_train=np.array([]), y_val=np.array([]), y_test=y[expected])

    config = make_config(csv_path, out)
    result = heldout_indices(config)
    assert np.array_equal(result, expected)


def test_raises_when_prepared_npz_does_not_match(tmp_path):
    csv_path, _ = _make_csv(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    np.savez_compressed(out / "prepared.npz", y_train=np.array([]), y_val=np.array([]), y_test=np.array([99, 99, 99]))

    config = make_config(csv_path, out)
    try:
        heldout_indices(config)
        assert False, "expected an AssertionError"
    except AssertionError as exc:
        assert "different split" in str(exc)
