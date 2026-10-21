import numpy as np

from tfacd.integrity.attacks import gaussian_noise, sign_flip, state_delta_l2_norm


def test_sign_flip_reverses_client_delta_around_global_state():
    global_state = {
        "weight": np.array([10.0, 2.0], dtype=np.float32),
        "num_batches_tracked": np.array(5, dtype=np.int64),
    }

    client_state = {
        "weight": np.array([12.0, 1.0], dtype=np.float32),
        "num_batches_tracked": np.array(6, dtype=np.int64),
    }

    poisoned = sign_flip(
        client_state,
        global_state,
        scale=2.0,
    )

    # delta = [12, 1] - [10, 2] = [2, -1]
    # poisoned = [10, 2] - 2 * [2, -1] = [6, 4]
    np.testing.assert_allclose(
        poisoned["weight"],
        np.array([6.0, 4.0], dtype=np.float32),
    )

    # Integer buffers remain valid and are not treated as signed updates.
    assert poisoned["num_batches_tracked"] == 6

    # Confirm the inputs were not mutated in place.
    np.testing.assert_array_equal(
        client_state["weight"],
        np.array([12.0, 1.0], dtype=np.float32),
    )

def test_gaussian_noise_is_relative_to_update_and_preserves_integer_buffers():
    global_state = {
        "weight": np.array([10.0, 20.0, -5.0, 1.0], dtype=np.float32),
        "num_batches_tracked": np.array(7, dtype=np.int64),
    }
    client_state = {
        "weight": np.array([11.0, 22.0, -4.0, 1.0], dtype=np.float32),
        "num_batches_tracked": np.array(8, dtype=np.int64),
    }

    poisoned_a = gaussian_noise(client_state, global_state, sigma=0.5, seed=123)
    poisoned_b = gaussian_noise(client_state, global_state, sigma=0.5, seed=123)
    np.testing.assert_array_equal(poisoned_a["weight"], poisoned_b["weight"])
    assert poisoned_a["num_batches_tracked"] == 8
    assert state_delta_l2_norm(poisoned_a, client_state) > 0.0
    assert state_delta_l2_norm(poisoned_a, global_state) > 0.0
    # The perturbation is calibrated against the actual update, not absolute weights.
    update_norm = state_delta_l2_norm(client_state, global_state)
    noise_norm = state_delta_l2_norm(poisoned_a, client_state)
    assert 0.05 < noise_norm / update_norm < 1.5
    np.testing.assert_array_equal(client_state["weight"], np.array([11.0, 22.0, -4.0, 1.0], dtype=np.float32))


def test_gaussian_noise_rejects_negative_strength_and_incompatible_states():
    state = {"weight": np.array([1.0], dtype=np.float32)}
    import pytest

    with pytest.raises(ValueError, match="finite and non-negative"):
        gaussian_noise(state, state, sigma=-0.1, seed=1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        gaussian_noise(state, state, sigma=float("nan"), seed=1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        gaussian_noise(state, state, sigma=float("inf"), seed=1)
    with pytest.raises(ValueError, match="identical keys"):
        gaussian_noise(state, {"other": state["weight"]}, sigma=0.1, seed=1)
