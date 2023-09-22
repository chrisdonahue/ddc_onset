import os
import pickle
import unittest

import numpy as np
import torch

from ddc.constants import CoarseDifficulty
from ddc.paths import (
    TEST_DATA_DIR,
    PARAMS_FEATURE_NORMALIZER_FP,
    PARAMS_PLACEMENT_CNN_FP,
)
from ddc.placement import *

_SHORT_FEATS_REF_FP = os.path.join(TEST_DATA_DIR, "short_essentia_feats_ref.npy")
_SHORT_SCORES_REF_FP = os.path.join(
    TEST_DATA_DIR, "short_essentia_feats_ref_ddc_placement_scores.pkl"
)
_LONG_FEATS_REF_FP = os.path.join(TEST_DATA_DIR, "long_essentia_feats_ref.npy")
_LONG_SCORES_REF_FP = os.path.join(
    TEST_DATA_DIR, "long_essentia_feats_ref_ddc_placement_scores.pkl"
)

_SHORT_DIFF_TO_NUM_PEAKS = {0: (69, 0), 1: (69, 0), 2: (68, 1), 3: (68, 3), 4: (66, 7)}

_LONG_DIFF_TO_NUM_PEAKS = {
    0: (3051, 718),
    1: (3033, 815),
    2: (3025, 942),
    3: (3027, 1170),
    4: (3031, 1288),
}


class TestPlacement(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.normalizer = FeatureNormalizer()
        self.normalizer.load_state_dict(torch.load(PARAMS_FEATURE_NORMALIZER_FP))
        self.normalizer.eval()
        self.normalizer.to(self.device)

        self.placement = PlacementCNN()
        self.placement.load_state_dict(torch.load(PARAMS_PLACEMENT_CNN_FP))
        self.placement.eval()
        self.placement.to(self.device)

    def test_placement(self):
        for feats_fp, scores_ref_fp, diff_to_num_peaks in zip(
            [_SHORT_FEATS_REF_FP, _LONG_FEATS_REF_FP],
            [_SHORT_SCORES_REF_FP, _LONG_SCORES_REF_FP],
            [_SHORT_DIFF_TO_NUM_PEAKS, _LONG_DIFF_TO_NUM_PEAKS],
        ):
            feats = np.load(feats_fp)
            feats = torch.tensor(feats, device=self.device, requires_grad=False)
            feats_norm = self.normalizer(feats)

            with open(scores_ref_fp, "rb") as f:
                diff_to_scores_ref = pickle.load(f, encoding="latin1")
            scores_ref = [diff_to_scores_ref[d.value] for d in CoarseDifficulty]
            scores_ref = np.array(scores_ref)

            for conv_chunk_size in [128, 256, 512]:
                for dense_chunk_size in [128, 256, 512]:
                    print(conv_chunk_size, dense_chunk_size)
                    with torch.no_grad():
                        diffs = torch.tensor(
                            [d.value for d in CoarseDifficulty],
                            device=self.device,
                            dtype=torch.int64,
                        )
                        scores = self.placement.forward(
                            feats_norm,
                            diffs,
                            conv_chunk_size=conv_chunk_size,
                            dense_chunk_size=dense_chunk_size,
                        )
                        scores = scores.cpu().numpy()

                    error = np.sum(np.abs(scores - scores_ref))
                    self.assertTrue(error < 1e-3)

                    for d in CoarseDifficulty:
                        peaks = find_peaks(scores[d.value])
                        thresholded_peaks = threshold_peaks(
                            scores[d.value], peaks, difficulty=d
                        )
                        self.assertEqual(
                            (len(peaks), len(thresholded_peaks)),
                            diff_to_num_peaks[d.value],
                        )


if __name__ == "__main__":
    unittest.main()
