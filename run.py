"""Source entry point for preprocessing, training and competition prediction."""
from pathlib import Path
import multiprocessing
import sys

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'_core')]

if __name__=='__main__':
    multiprocessing.freeze_support()
    from eatdet.cli import main
    raise SystemExit(main())
