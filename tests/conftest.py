"""Make the example setups (examples/setups) importable in the tests: they are example data, not part of SwingPy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
