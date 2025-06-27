from pathlib import Path
from typing import Final

ROOT_DIR: Final = Path(__file__).parent.parent.parent
INPUT_DIR = ROOT_DIR / "input"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
