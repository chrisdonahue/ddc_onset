from .paths import LIB_DIR
from .constants import SAMPLE_RATE, FRAME_RATE, Difficulty
from .spectral import SpectrogramExtractor
from .cnn import SpectrogramNormalizer, PlacementCNN
from .util import compute_onset_salience, find_peaks, threshold_peaks


# NOTE: This changes the test discovery pattern from "test*.py" (default) to "*test.py".
def load_tests(loader, standard_tests, pattern):
    package_tests = loader.discover(start_dir=LIB_DIR, pattern="*test.py")
    standard_tests.addTests(package_tests)
    return standard_tests
