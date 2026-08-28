import numpy as np

from tfacd.data.sequences import as_model_input, make_sequences


def test_sequence_windowing():
    x = np.arange(30, dtype=np.float32).reshape(10, 3)
    y = np.arange(10)
    sx, sy = make_sequences(x, y, sequence_length=4, stride=2)
    assert sx.shape == (4, 4, 3)
    assert sy.tolist() == [3, 5, 7, 9]


def test_as_model_input_passthrough_3d():
    x = np.ones((5, 8, 4), dtype=np.float32)
    y = np.arange(5)
    sx, sy = as_model_input(x, y, sequence_length=8, stride=1)
    assert sx.shape == (5, 8, 4)
    assert np.shares_memory(sx, x) or np.array_equal(sx, x)
    assert sy.tolist() == y.tolist()
