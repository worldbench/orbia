"""Download workflow weights once and write a configuration with local paths."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
URLS = {
    'dinov2_vits14_pretrain.pth': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
    'alex.pth': 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/alex.pth',
    'torch-home/hub/checkpoints/alexnet-owt-7be5be79.pth': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
    'musiq.pth': 'https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/musiq_spaq_ckpt-358bb6af.pth',
    'aesthetic.pth': 'https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth',
    'ViT-L-14.pt': 'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt',
    'amt-s.pth': 'https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth',
    'salad.ckpt': 'https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt',
}


def download_url(url, target):
    if target.is_file() and target.stat().st_size:
        print(f'Reusing {target.name}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name+'.part')
    print(f'Downloading {target.name}')
    request = urllib.request.Request(url, headers={'User-Agent': 'ORBIA-setup'})
    with urllib.request.urlopen(request, timeout=120) as source, partial.open('wb') as output:
        shutil.copyfileobj(source, output, length=1024*1024)
    partial.replace(target)


def download_models(workflow, weights):
    from huggingface_hub import hf_hub_download, snapshot_download
    # Resolve gated SAM access before downloading the other large models.
    # Hugging Face uses the caller's existing login; tokens are never printed.
    hf_hub_download('facebook/sam3.1', 'sam3.1_multiplex.pt', local_dir=str(weights))
    if workflow == 'evaluation':
        for name, url in URLS.items():
            download_url(url, weights/name)
    snapshots = [('depth-anything/DA3-GIANT-1.1', 'DA3-GIANT-1.1')]
    snapshots.append(('Qwen/Qwen2-VL-7B-Instruct', 'Qwen2-VL-7B-Instruct')
                     if workflow == 'evaluation' else ('Qwen/Qwen3.5-9B', 'Qwen3.5-9B'))
    for repository, name in snapshots:
        print(f'Downloading or reusing {repository}')
        snapshot_download(repository, local_dir=str(weights/name),
            allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.py'])
    if workflow == 'evaluation':
        hf_hub_download('MizzenAI/HPSv3', 'HPSv3.safetensors', local_dir=str(weights))


def evaluation_config(source, output, sources, weights):
    import yaml
    cfg = yaml.safe_load(source.read_text())
    evaluator = cfg['evaluator']
    evaluator.update(torch_home=str(weights/'torch-home'), dino_repo=str(sources/'dinov2'),
        dino_checkpoint=str(weights/'dinov2_vits14_pretrain.pth'),
        lpips_checkpoint=str(weights/'alex.pth'), musiq_checkpoint=str(weights/'musiq.pth'),
        aesthetic_clip_checkpoint=str(weights/'ViT-L-14.pt'),
        aesthetic_checkpoint=str(weights/'aesthetic.pth'),
        vbench_root=str(sources/'VBench'),
        amt_config=str(sources/'VBench/vbench/third_party/amt/cfgs/AMT-S.yaml'),
        amt_checkpoint=str(weights/'amt-s.pth'),
        da3_python=None,
        da3_streaming_python=None,
        da3_repo=str(sources/'Depth-Anything-3'),
        da3_streaming_repo=str(sources/'Depth-Anything-3'),
        da3_checkpoint=str(weights/'DA3-GIANT-1.1'),
        da3_streaming_checkpoint=str(weights/'DA3-GIANT-1.1'),
        da3_streaming_salad_checkpoint=str(weights/'salad.ckpt'),
        anchor_sam_python=None,
        anchor_sam_repo=str(sources/'sam3'),
        anchor_sam_checkpoint=str(weights/'sam3.1_multiplex.pt'))
    hps_path = weights/'HPSv3_7B.local.yaml'
    hps_template = sources/'HPSv3/hpsv3/config/HPSv3_7B.yaml'
    if not hps_template.is_file():
        raise FileNotFoundError(f'{hps_template}: run install_evaluation.sh first')
    hps = yaml.safe_load(hps_template.read_text())
    hps.update(model_name_or_path=str(weights/'Qwen2-VL-7B-Instruct'),
        output_dir=str(ROOT/'cache/hps'), deepspeed=None, report_to='none',
        disable_flash_attn2=True)
    hps_path.parent.mkdir(parents=True, exist_ok=True)
    hps_path.write_text(yaml.safe_dump(hps, sort_keys=False))
    cfg.setdefault('quality', {}).update(hps_repo=str(sources/'HPSv3'),
        hps_config=str(hps_path), hps_checkpoint=str(weights/'HPSv3.safetensors'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False))


def construction_config(source, output, sources, weights):
    cfg = json.loads(source.read_text())
    cfg.setdefault('models', {}).setdefault('vlm', {}).update(model=str(weights/'Qwen3.5-9B'))
    cfg['models'].setdefault('sam3', {}).update(repository=str(sources/'sam3'),
        checkpoint=str(weights/'sam3.1_multiplex.pt'))
    cfg.setdefault('geometry', {}).update(checkpoint=str(weights/'DA3-GIANT-1.1'))
    # Relative auxiliary files (e.g. pose_gates.json) remain valid beside the template.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cfg, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', choices=('evaluation', 'construction'), required=True)
    parser.add_argument('--source-dir', type=Path, default=ROOT/'third_party')
    parser.add_argument('--weights-dir', type=Path, default=ROOT/'weights')
    parser.add_argument('--config', type=Path, help='existing workflow template to copy')
    parser.add_argument('--output-config', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='list downloads and output without network access or writes')
    parser.add_argument('--config-only', action='store_true', help='write paths for already downloaded weights')
    args = parser.parse_args()
    source = (args.config or ROOT/'configs'/('evaluation.yaml' if args.workflow == 'evaluation'
                                            else 'construction.json')).resolve()
    output = (args.output_config or source.with_name(source.stem+'.local'+source.suffix)).resolve()
    if source == output:
        parser.error('output configuration must differ from the input template')
    sources, weights = [p.expanduser().resolve() for p in
                       (args.source_dir, args.weights_dir)]
    if args.dry_run:
        print(json.dumps(dict(workflow=args.workflow, source=str(source), output=str(output),
            environment='orbia-eval' if args.workflow == 'evaluation' else 'orbia-construction',
            python=sys.executable,
            repositories=['depth-anything/DA3-GIANT-1.1', 'facebook/sam3.1',
                'Qwen/Qwen2-VL-7B-Instruct' if args.workflow == 'evaluation' else 'Qwen/Qwen3.5-9B']
                + (['MizzenAI/HPSv3'] if args.workflow == 'evaluation' else []),
            url_downloads=URLS if args.workflow == 'evaluation' else {},
            weights_directory=str(weights)), indent=2))
        return 0
    if not args.config_only:
        download_models(args.workflow, weights)
    if args.workflow == 'evaluation':
        evaluation_config(source, output, sources, weights)
    else:
        construction_config(source, output, sources, weights)
    print(f'Wrote {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
