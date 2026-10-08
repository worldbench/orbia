"""Pack a dataset folder into public/references archives, and extract them again."""
from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path


def _reference_files(document):
    files = [f[k] for f in document['frames'].values() for k in ('rgb', 'depth', 'valid')]
    files += [a[k] for a in document['anchors'] for k in ('q0_mask', 'qe_mask', 'qe_region')]
    files += list(document['observability'].values()) + [document['qe_scene_region']]
    return files


def package_dataset(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError('output must be outside the source dataset')
    ids = [row['case_id'] for row in json.loads((source/'manifest.json').read_text())['cases']]
    if len(set(ids)) != len(ids):
        raise ValueError('manifest contains duplicate cases')
    output.mkdir(parents=True, exist_ok=True)
    for folder in ('public', 'references'):
        temporary = output/(folder+'.tar.gz.partial')
        with tarfile.open(temporary, 'w:gz', compresslevel=1, dereference=True) as archive:
            for cid in ids:
                root = source/folder/cid
                if folder == 'references':
                    document = json.loads((root/'reference.json').read_text())
                    for name in _reference_files(document):
                        if Path(name).is_absolute() or '..' in Path(name).parts or not (root/name).is_file():
                            raise ValueError(f'missing or invalid asset: {cid}/{name}')
                archive.add(root, arcname=f'{folder}/{cid}')
        temporary.replace(output/(folder+'.tar.gz'))
    shutil.copyfile(source/'manifest.json', output/'manifest.json')
    return {'cases': len(ids), 'archives': {name: (output/name).stat().st_size
                                            for name in ('public.tar.gz', 'references.tar.gz')}}


def extract_dataset(archives, output, *, references=True):
    archives, output = Path(archives), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ('public', 'references') if references else ('public',):
        with tarfile.open(archives/(name+'.tar.gz'), 'r:gz') as archive:
            for member in archive:
                target = output/member.name
                if not target.resolve().is_relative_to(output) or not (member.isdir() or member.isfile()):
                    raise ValueError('unsafe archive member: '+member.name)
                archive.extract(member, output)
    shutil.copyfile(archives/'manifest.json', output/'manifest.json')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package_dataset(args.source, args.output), indent=2))
