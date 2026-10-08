"""Check Python dependencies and configured model files before running a workflow."""
import argparse
import importlib
import importlib.util
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', choices=('loader', 'construction', 'scannetpp', 'evaluation', 'ue', 'models'), default='loader')
    parser.add_argument('--config', type=Path, help='optional evaluation YAML to check source and weight paths')
    parser.add_argument('--cuda', action='store_true', help='import evaluation backends and run small CUDA operations; no weights needed')
    args = parser.parse_args()
    required = ['numpy', 'PIL']
    required += {'loader': [], 'construction': ['scipy', 'cv2'],
                 'evaluation': ['scipy', 'cv2', 'yaml', 'torch', 'torchvision', 'lpips', 'pyiqa', 'clip',
                                'sam3', 'depth_anything_3', 'hpsv3', 'gsplat', 'open3d', 'xformers'],
                 'scannetpp': ['scipy', 'cv2', 'open3d'],
                 'ue': ['scipy', 'cv2', 'OpenImageIO'],
                 'models': ['scipy', 'cv2', 'imageio']}[args.workflow]
    failures = []
    for module in required:
        found = importlib.util.find_spec(module) is not None
        print(f'{module}: '+('found' if found else 'missing'))
        if not found:
            failures.append(module)
    for package in ('numpy', 'Pillow', 'torch', 'transformers'):
        try:
            print(f'{package} version: {version(package)}')
        except PackageNotFoundError:
            pass
    if args.cuda and not failures:
        backends = (
            ('sam3.model_builder', 'build_sam3_multiplex_video_predictor'),
            ('depth_anything_3.api', 'DepthAnything3'),
            ('hpsv3.inference', 'HPSv3RewardInferencer'),
            ('gsplat', 'rasterization'),
        ) if args.workflow == 'evaluation' else ()
        for module, symbol in backends:
            try:
                getattr(importlib.import_module(module), symbol)
                print(f'{module}.{symbol}: imported')
            except Exception as error:
                print(f'{module}: {type(error).__name__}: {error}', file=sys.stderr)
                failures.append(module)
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable')
            matrix = torch.randn(128, 128, device='cuda')
            if not torch.isfinite(matrix @ matrix).all().item():
                raise RuntimeError('CUDA matrix multiplication produced non-finite values')
            if args.workflow == 'evaluation':
                from xformers.ops import memory_efficient_attention
                q = torch.randn(1, 64, 2, 64, device='cuda', dtype=torch.float16)
                if not torch.isfinite(memory_efficient_attention(q, q, q)).all().item():
                    raise RuntimeError('xFormers attention produced non-finite values')
            torch.cuda.synchronize()
            print(f'CUDA: passed on {torch.cuda.get_device_name(0)} (runtime {torch.version.cuda})')
        except Exception as error:
            print(f'CUDA: {type(error).__name__}: {error}', file=sys.stderr)
            failures.append('CUDA')
    if args.config:
        import yaml
        config = yaml.safe_load(args.config.read_text())
        for section in ('evaluator', 'quality'):
            for key, value in config.get(section, {}).items():
                if isinstance(value, str) and key.endswith(('_repo', '_root', '_checkpoint', '_config', '_python')):
                    path = Path(value).expanduser()
                    found = path.exists()
                    print(f'{key}: '+('found' if found else 'missing')+f' ({path})')
                    if not found:
                        failures.append(key)
    if failures:
        print('Missing requirements: '+', '.join(failures), file=sys.stderr)
        return 1
    print('Selected workflow requirements found.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
