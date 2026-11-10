"""Multimodal eating-event detection."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT/'_core') not in sys.path:sys.path.insert(0,str(ROOT/'_core'))
