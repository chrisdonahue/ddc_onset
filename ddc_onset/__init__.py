from .spectral import SpectrogramExtractor
from .cnn import SpectrogramNormalizer, PlacementCNN
from .constants import FRAME_RATE
from .paths import LIB_DIR


# NOTE: This changes the test discovery pattern from "test*.py" (default) to "*test.py".
def load_tests(loader, standard_tests, pattern):
    package_tests = loader.discover(start_dir=LIB_DIR, pattern="*test.py")
    standard_tests.addTests(package_tests)
    return standard_tests
