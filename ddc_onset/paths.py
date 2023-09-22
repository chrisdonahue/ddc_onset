import pathlib

LIB_DIR = pathlib.Path(__file__).resolve().parent
_REPO_DIR = LIB_DIR.parent if LIB_DIR.parent.name == "ddc_onset" else None

WEIGHTS_DIR = pathlib.Path(LIB_DIR, "weights")
TEST_DATA_DIR = None if _REPO_DIR is None else pathlib.Path(_REPO_DIR, "test")
