"""Inspect a benchmark input, requested cameras, and optional references."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from src.data import load_case  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--references', action='store_true')
    args = parser.parse_args()
    case = load_case(args.dataset, args.case_id, references=args.references)
    result = dict(case_id=case.case_id, template=case.template, prompt=case.prompt,
                  input_shape=list(case.q0_rgb.shape), camera_shape=list(case.poses_c2w.shape),
                  pose_unit=case.document['pose_unit'], events=case.document['events'],
                  anchors=list(case.anchor_masks))
    print(json.dumps(result, indent=2))
