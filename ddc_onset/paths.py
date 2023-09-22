import pathlib

LIB_DIR = pathlib.Path(__file__).resolve().parent
_REPO_DIR = None
if LIB_DIR.parent.name == "ddc_onset":
    _REPO_DIR = LIB_DIR.parent

WEIGHTS_DIR = pathlib.Path(LIB_DIR, "weights")
TEST_DATA_DIR = None if _REPO_DIR is None else pathlib.Path(_REPO_DIR, "test")


# NOTE: This changes the test discovery pattern from "test*.py" (default) to "*test.py".
def load_tests(loader, standard_tests, pattern):
    package_tests = loader.discover(start_dir=LIB_DIR, pattern="*test.py")
    standard_tests.addTests(package_tests)
    return standard_tests
