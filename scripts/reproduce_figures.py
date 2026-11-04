"""CPU-only reproduction of the seven figures into a separate directory."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_publication.figures import reproduce_figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced_figures")
    parser.add_argument("--png-only", action="store_true")
    args = parser.parse_args()
    for path in reproduce_figures(root=ROOT, output=args.output,
                                  formats=("png",) if args.png_only else ("pdf", "png")):
        print(path)
