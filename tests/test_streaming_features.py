import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tfacd.streaming.features import StreamingFeatureExtractor

NUMERIC_COLUMNS = ["num1", "num2"]
CATEGORICAL_COLUMNS = ["cat1"]
INPUT_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def _fit_transformer():
    train = pd.DataFrame(
        {"num1": [1.0, 2.0, 3.0, 4.0], "num2": [10.0, 20.0, 30.0, 40.0], "cat1": ["a", "b", "a", "c"]}
    )
    transformer = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_COLUMNS),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    transformer.fit(train)
    return transformer


@pytest.fixture
def extractor(tmp_path):
    transformer = _fit_transformer()
    joblib.dump(transformer, tmp_path / "preprocessor.joblib")
    feature_dim = transformer.transform(pd.DataFrame({"num1": [1.0], "num2": [1.0], "cat1": ["a"]})).shape[1]
    metadata = {
        "input_columns": INPUT_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "feature_dim": feature_dim,
        "num_classes": 2,
        "classes": ["Normal", "Attack"],
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return StreamingFeatureExtractor(tmp_path), transformer


def test_matches_transformer_own_transform_directly(extractor):
    ext, transformer = extractor
    records = [{"num1": "2.0", "num2": "20.0", "cat1": "b", "irrelevant_column": "x"}]
    expected = transformer.transform(pd.DataFrame({"num1": [2.0], "num2": [20.0], "cat1": ["b"]}))
    actual = ext.transform(records)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_string_valued_numeric_columns_are_coerced(extractor):
    ext, _ = extractor
    records = [{"num1": "3.0", "num2": "30.0", "cat1": "a"}]
    result = ext.transform(records)
    assert result.shape[1] > 0  # ran without raising, values are usable floats
    assert np.isfinite(result).all()


def test_categorical_arriving_as_int_vs_str_produces_different_output(extractor):
    """Locks down the corruption finding: if a chunked read hands a categorical
    column back as a numeric dtype, the fitted OneHotEncoder must not silently
    accept it as if it matched a known string category."""
    ext, _ = extractor
    as_str = ext.transform([{"num1": "1.0", "num2": "10.0", "cat1": "a"}])
    as_int = ext.transform([{"num1": "1.0", "num2": "10.0", "cat1": 0}])  # int, not the string "a"
    assert not np.allclose(as_str, as_int)


def test_missing_column_raises(extractor):
    ext, _ = extractor
    with pytest.raises(KeyError):
        ext.transform([{"num1": "1.0", "cat1": "a"}])  # num2 missing


def test_empty_records_returns_correctly_shaped_empty_array(extractor):
    ext, _ = extractor
    result = ext.transform([])
    assert result.shape == (0, ext.metadata["feature_dim"])


def test_transform_never_refits_categories(extractor):
    ext, transformer = extractor
    onehot = transformer.named_transformers_["categorical"].named_steps["onehot"]
    before = [c.copy() for c in onehot.categories_]
    ext.transform([{"num1": "1.0", "num2": "10.0", "cat1": "totally-unseen-category"}])
    after = onehot.categories_
    for b, a in zip(before, after):
        np.testing.assert_array_equal(b, a)
