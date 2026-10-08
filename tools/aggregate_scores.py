"""Aggregate completed case reports into T1–T6 scores and model ranks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from src.evaluate import aggregate_directory  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='evaluation output directory')
    parser.add_argument('--output', required=True, help='destination for JSON and CSV scores')
    args = parser.parse_args()
    aggregate_directory(args.input, args.output)
