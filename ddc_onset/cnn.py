import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import Difficulty, DIFFICULTY_TO_PLACEMENT_THRESHOLD

_NUM_MEL_BANDS = 80
_NUM_FFT_FRAME_LENGTHS = 3


class SpectrogramNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(
            torch.zeros(
                (_NUM_MEL_BANDS, _NUM_FFT_FRAME_LENGTHS),
                dtype=torch.float32,
                requires_grad=False,
            ),
            requires_grad=False,
        )
        self.std = nn.Parameter(
            torch.ones(
                (_NUM_MEL_BANDS, _NUM_FFT_FRAME_LENGTHS),
                dtype=torch.float32,
                requires_grad=False,
            ),
            requires_grad=False,
        )

    def forward(self, x):
        return (x - self.mean) / self.std


_FEATURE_CONTEXT_RADIUS_1 = 7
_FEATURE_CONTEXT_RADIUS_2 = 3


class PlacementCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 10, (7, 3))
        self.maxpool0 = nn.MaxPool2d((1, 3), (1, 3))
        self.conv1 = nn.Conv2d(10, 20, (3, 3))
        self.maxpool1 = nn.MaxPool2d((1, 3), (1, 3))
        self.dense0 = nn.Linear(1125, 256)
        self.dense1 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)

    def conv(self, x):
        # x is b, 3, 15, 80

        # Conv 0
        x = self.conv0(x)
        x = F.relu(x)
        x = self.maxpool0(x)

        # Conv 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        return x

    def dense(self, x_conv_diff):
        # x is b, 1125

        # Dense 0
        x = self.dense0(x_conv_diff)
        x = F.relu(x)

        # Dense 1
        x = self.dense1(x)
        x = F.relu(x)

        # Output
        x = self.output(x)
        x = x.view(-1)

        return x

    def forward(
        self, x, ds, conv_chunk_size=256, dense_chunk_size=256, output_logits=False
    ):
        # x is t, 80, 3
        num_timesteps = x.shape[0]

        # Pad features
        x_padded = F.pad(
            x, (0, 0, 0, 0, _FEATURE_CONTEXT_RADIUS_1, _FEATURE_CONTEXT_RADIUS_1)
        )

        # Convolve
        x_padded = x_padded.permute(2, 0, 1)
        x_conv = []
        for i in range(0, num_timesteps, conv_chunk_size):
            x_chunk = x_padded[
                :, i : i + conv_chunk_size + _FEATURE_CONTEXT_RADIUS_1 * 2
            ].unsqueeze(0)
            x_chunk_conv = self.conv(x_chunk)
            assert x_chunk_conv.shape[1] > _FEATURE_CONTEXT_RADIUS_2 * 2
            if i == 0:
                x_conv.append(x_chunk_conv[:, :, :_FEATURE_CONTEXT_RADIUS_2])
            x_conv.append(
                x_chunk_conv[:, :, _FEATURE_CONTEXT_RADIUS_2:-_FEATURE_CONTEXT_RADIUS_2]
            )
        x_conv.append(x_chunk_conv[:, :, -_FEATURE_CONTEXT_RADIUS_2:])
        x_conv = torch.cat(x_conv, dim=2)
        x_conv = x_conv.permute(0, 2, 3, 1)
        x_conv = x_conv.reshape(-1, 160)

        # Dense
        logits = []
        for i in range(0, num_timesteps, dense_chunk_size):
            # TODO: Turn this into a convolutional layer?
            # NOTE: Pytorch didn't like this as of 20_03_15:
            # https://github.com/pytorch/pytorch/pull/33073
            x_chunk = []
            for j in range(i, i + dense_chunk_size):
                if j >= num_timesteps:
                    break
                x_chunk.append(x_conv[j : j + 1 + _FEATURE_CONTEXT_RADIUS_2 * 2])
            x_chunk = torch.stack(x_chunk, dim=0)
            x_chunk = x_chunk.reshape(-1, 1120)

            # Compute dense layer for each difficulty
            logits_diffs = []
            for k in range(ds.shape[0]):
                d = ds[k].repeat(x_chunk.shape[0])
                doh = F.one_hot(d, len(Difficulty)).float()
                x_chunk_diff = torch.cat([x_chunk, doh], dim=1)
                x_chunk_dense = self.dense(x_chunk_diff)
                logits_diffs.append(x_chunk_dense)
            logits_diffs = torch.stack(logits_diffs, dim=0)
            logits.append(logits_diffs)
        logits = torch.cat(logits, dim=1)

        if output_logits:
            return logits
        else:
            return torch.sigmoid(logits)


def find_peaks(scores):
    try:
        from scipy.signal import argrelextrema
    except ImportError:
        raise Exception(
            'Scipy required for finding peaks. Please install scipy with "pip install scipy"'
        )
    scores_smoothed = np.convolve(scores, np.hamming(5), "same")
    peaks = argrelextrema(scores_smoothed, np.greater_equal, order=1)[0]
    return peaks


def threshold_peaks(scores, peaks, difficulty=None, threshold=None):
    if difficulty is None and threshold is None:
        raise ValueError("Must specify either difficulty or threshold")
    if threshold is None:
        threshold = COARSE_DIFFICULTY_TO_PLACEMENT_THRESHOLD[difficulty]
    return [i for i in peaks if scores[i] >= threshold]
