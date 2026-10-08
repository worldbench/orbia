"""Check a submission folder before sending it: metadata, case coverage and frame counts."""
import argparse
import json
from pathlib import Path

TYPES = ('camera', 'action', 'text')


def frame_count(video):
    import cv2
    capture = cv2.VideoCapture(str(video))
    fps, count = capture.get(cv2.CAP_PROP_FPS), 0
    while capture.grab():
        count += 1
    capture.release()
    return count, fps


def check(submission, dataset):
    submission, dataset = Path(submission), Path(dataset)
    issues, notes = [], []
    meta_path = submission/'meta.json'
    if not meta_path.is_file():
        return ['missing meta.json'], notes
    meta = json.loads(meta_path.read_text())
    for field in ('model_name', 'display_name', 'type', 'org', 'contact'):
        if not meta.get(field):
            issues.append(f'meta.json: missing {field}')
    if meta.get('type') not in TYPES:
        issues.append(f'meta.json: type must be one of {", ".join(TYPES)}')
    tags = meta.get('tags', [])
    if len(tags) > 4 or any(not 1 <= len(str(t)) <= 16 for t in tags):
        issues.append('meta.json: use at most 4 tags of 1-16 characters')

    if not (dataset/'public').is_dir():
        return issues + ['dataset/public is missing; pass --dataset with the ORBIA dataset'], notes
    cases = {p.name: json.loads((p/'case.json').read_text())['frame_count']
             for p in sorted((dataset/'public').iterdir()) if (p/'case.json').is_file()}
    if not cases:
        return issues + ['dataset/public contains no cases'], notes
    videos = {p.stem: p for p in (submission/'videos').glob('*.mp4')}
    missing, extra = sorted(set(cases) - set(videos)), sorted(set(videos) - set(cases))
    if missing:
        issues.append(f'{len(missing)} missing videos, e.g. {", ".join(missing[:5])}')
    if extra:
        issues.append(f'{len(extra)} videos with unknown case IDs, e.g. {", ".join(extra[:5])}')
    for case_id in sorted(set(cases) & set(videos)):
        count, fps = frame_count(videos[case_id])
        if count != cases[case_id]:
            issues.append(f'{case_id}: {count} frames, expected {cases[case_id]}')
        if round(fps) != 16:
            notes.append(f'{case_id}: saved at {fps:g} FPS (16 recommended; frames are matched by index)')

    scores = submission/'results'/'scores'/'scores.json'
    if (submission/'results').exists() and not scores.is_file():
        issues.append('results/ is present but results/scores/scores.json is missing')
    return issues, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission', help='submission folder containing meta.json and videos/')
    parser.add_argument('--dataset', default='data/orbia', help='dataset root (default: data/orbia)')
    args = parser.parse_args()
    issues, notes = check(args.submission, args.dataset)
    for note in notes[:5]:
        print('note:', note)
    if len(notes) > 5:
        print(f'note: {len(notes) - 5} more videos not saved at 16 FPS')
    for issue in issues:
        print('x', issue)
    print('Submission looks good.' if not issues else f'{len(issues)} issue(s) found.')
    raise SystemExit(1 if issues else 0)


if __name__ == '__main__':
    main()
