from unittest.mock import patch

from tfacd.federated.loaders import _load_prepared_cached, client_loaders


def test_load_prepared_cached_only_calls_load_prepared_once_per_output_dir():
    _load_prepared_cached.cache_clear()
    with patch("tfacd.federated.loaders.load_prepared") as mocked:
        mocked.return_value = "sentinel-prepared"
        first = _load_prepared_cached("artifacts/data")
        second = _load_prepared_cached("artifacts/data")

    assert first == "sentinel-prepared"
    assert second == "sentinel-prepared"
    mocked.assert_called_once_with("artifacts/data")


def test_load_prepared_cached_distinguishes_different_output_dirs():
    _load_prepared_cached.cache_clear()
    with patch("tfacd.federated.loaders.load_prepared") as mocked:
        mocked.side_effect = lambda output_dir: f"prepared-for-{output_dir}"
        a = _load_prepared_cached("dir-a")
        b = _load_prepared_cached("dir-b")

    assert a == "prepared-for-dir-a"
    assert b == "prepared-for-dir-b"
    assert mocked.call_count == 2


def test_client_loaders_works_against_real_artifacts_and_reuses_cache():
    """Real end-to-end check: two different clients' loaders must not
    cross-contaminate despite sharing the same cached `prepared` object -
    fancy indexing (x[indices]) always copies, never views."""
    _load_prepared_cached.cache_clear()
    from tfacd.common.config import load_config

    config = load_config("configs/edge_iiot.yaml")
    train0, val0 = client_loaders(config, client_id=0, batch_size=64)
    train1, val1 = client_loaders(config, client_id=1, batch_size=64)

    assert _load_prepared_cached.cache_info().hits >= 1  # second client_loaders() call reused the cached load
    assert len(train0.dataset) > 0
    assert len(train1.dataset) > 0
    # Different clients' partitions must not be identical (Dirichlet-partitioned, disjoint indices).
    assert not (train0.dataset.x.shape == train1.dataset.x.shape and (train0.dataset.x == train1.dataset.x).all())
