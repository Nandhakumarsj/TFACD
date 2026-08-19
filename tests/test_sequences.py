import numpy as np

from tfacd.data.sequences import make_sequences


def test_sequence_windowing():
    x = np.arange(30, dtype=np.float32).reshape(10, 3)
    y = np.arange(10)
    sx, sy = make_sequences(x, y, sequence_length=4, stride=2)
    assert sx.shape == (4, 4, 3)
    assert sy.tolist() == [3, 5, 7, 9]
