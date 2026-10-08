"""Generate benchmark videos with a registered or user-defined world model."""
import argparse
import json

from src.models import generate, get_model, merge_manifests, registered_models


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', help=f'registered name {registered_models()} or package.module:Class')
    parser.add_argument('--model-args', default='{}', help='JSON keyword arguments for the model constructor')
    parser.add_argument('--name', help='model name written to the evaluation manifest')
    parser.add_argument('--dataset', help='extracted dataset root containing public/')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cases', nargs='+', help='case IDs (default: all)')
    parser.add_argument('--horizon', choices=('short', 'long'))
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--merge', action='store_true', help='only merge videos.shard*.jsonl')
    args = parser.parse_args()
    if args.merge:
        print(merge_manifests(args.output))
        return 0
    if not args.model or not args.dataset:
        parser.error('--model and --dataset are required')
    model = get_model(args.model, **json.loads(args.model_args))
    summary = generate(model, args.dataset, args.output, case_ids=args.cases, horizon=args.horizon,
                       shard=args.shard, num_shards=args.num_shards, overwrite=args.overwrite,
                       model_name=args.name)
    print(json.dumps(summary, indent=2))
    return 1 if summary['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
