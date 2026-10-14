import numpy as np

from tfacd.integrity.attacks import sign_flip


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