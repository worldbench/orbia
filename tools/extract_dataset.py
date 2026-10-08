"""Extract the existing public-input and evaluation-reference archives."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from src.data.bundle import extract_dataset  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--public-only', action='store_true')
    args = parser.parse_args()
    extract_dataset(args.archives, args.output, references=not args.public_only)
