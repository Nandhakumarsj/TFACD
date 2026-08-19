import torch

from tfacd.models.cnn_bilstm import CNNBiLSTM


def test_model_output_shape():
    model = CNNBiLSTM(feature_dim=32, num_classes=5, conv_channels=(8, 16), pooled_features=4, lstm_hidden=12)
    logits = model(torch.randn(7, 3, 32))
    assert logits.shape == (7, 5)
