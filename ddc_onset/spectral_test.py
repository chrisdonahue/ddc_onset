import os
import unittest

import numpy as np
import torch

from .paths import TEST_DATA_DIR, PARAMS_FEATURE_EXTRACTOR_FP
from .spectral import SpectrogramExtractor

_SHORT_AUDIO_REF_FP = os.path.join(TEST_DATA_DIR, "short_essentia_audio_ref.npy")
_SHORT_FEATS_REF_FP = os.path.join(TEST_DATA_DIR, "short_essentia_feats_ref.npy")
_LONG_AUDIO_REF_FP = os.path.join(TEST_DATA_DIR, "long_essentia_audio_ref.npy")
_LONG_FEATS_REF_FP = os.path.join(TEST_DATA_DIR, "long_essentia_feats_ref.npy")


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor()
        self.extractor.load_state_dict(torch.load(PARAMS_FEATURE_EXTRACTOR_FP))
        self.extractor.eval()
        self.extractor.to(self.device)

    def test_extract_feats(self):
        for frame_chunk_size in [None, 100, 10, 1]:
            for _AUDIO_REF_FP, _FEATS_REF_FP in zip(
                [_SHORT_AUDIO_REF_FP, _LONG_AUDIO_REF_FP],
                [_SHORT_FEATS_REF_FP, _LONG_FEATS_REF_FP],
            ):
                audio = np.load(_AUDIO_REF_FP).reshape(-1, 1)
                audio = torch.tensor(audio, requires_grad=False, device=self.device)

                feats_ref = np.load(_FEATS_REF_FP)
                with torch.no_grad():
                    feats = (
                        self.extractor(audio, frame_chunk_size=frame_chunk_size)
                        .cpu()
                        .numpy()
                    )

                # TODO: Fix shape here... How many frames do we expect 9585510 samples to have?
                self.assertTrue(feats.shape[0] >= feats_ref.shape[0])

                error = np.mean(np.abs(feats[: feats_ref.shape[0]] - feats_ref))
                self.assertTrue(error < 0.012)


if __name__ == "__main__":
    unittest.main()
