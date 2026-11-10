"""PPG-only meal-event detection. No IMU inputs are consumed by this package."""

__version__ = "1.0.0"
"""Training helpers use the same expert implementation as deployment."""
from pathlib import Path
import sys
_runtime = str(Path(__file__).resolve().parents[2] / "runtime")
if _runtime not in sys.path:
    sys.path.insert(0, _runtime)
