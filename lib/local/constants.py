import os
from pathlib import Path

EXECUTION_DIR = Path(os.getcwd())
WORKSPACE_ROOT = Path("/".join(os.path.realpath(__file__).split('/')[:-3]))

CACHE = EXECUTION_DIR.joinpath('cache')
DATA = WORKSPACE_ROOT.joinpath('data')
