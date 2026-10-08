"""Select, annotate, review, and export real-video benchmark cases."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.construction.pipeline import main  # noqa: E402

if __name__ == '__main__':
    main()
