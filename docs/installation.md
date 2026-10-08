# Installation

ORBIA uses separate Conda environments for evaluation, real-data construction and UE conversion.

| Environment | Used for | Install |
| --- | --- | --- |
| `orbia-eval` | Evaluation | `bash tools/install_evaluation.sh` |
| `orbia-construction` | Real-data construction | `bash tools/install_construction.sh` |
| `orbia-ue` | UE capture conversion (CPU only) | `bash tools/install_ue.sh` |

The GPU environments use Python 3.12, PyTorch 2.10 and CUDA 12.8. Evaluation compiles gsplat, so `nvcc` and a C++ compiler are needed. 

## Evaluation

```bash
bash tools/install_evaluation.sh
conda activate orbia-eval
python tools/download_weights.py --workflow evaluation
python tools/verify_install.py --workflow evaluation --cuda   # optional check
```

The install script clones upstream code (DA3, SAM 3.1, HPSv3, VBench) to `third_party/`. `download_weights.py` downloads the checkpoints to `weights/`.

## Construction

```bash
bash tools/install_construction.sh
conda activate orbia-construction
python tools/download_weights.py --workflow construction --config configs/construction/spatialvid.json
```

This writes `configs/construction/spatialvid.local.json` with local model paths. Use the config of your source (`sekai.json`, `mira.json`, `scannetpp.json`, `self_captured.json`) instead of `spatialvid.json` as needed.

## UE

```bash
bash tools/install_ue.sh
conda activate orbia-ue
```

Unreal Editor and its plugins are set up separately, see [unreal/README.md](../unreal/README.md).

## Other options

- `--name` changes the environment name, e.g. `bash tools/install_evaluation.sh --name my-env`.
- `--source-dir` and `--weights-dir` change where code and weights are stored.
- `download_weights.py --config-only` only writes the config, for weights you already have.
- To run your own model, install `requirements/models.txt` in the model's environment (see [models.md](models.md)).

## Models and weights

| Component | Used for | Checkpoint |
| --- | --- | --- |
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Camera/depth recovery, Gaussian reconstruction | DA3-GIANT-1.1 |
| [DA3 Streaming](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/main/da3_streaming) + [SALAD](https://github.com/serizba/salad) | Long-video geometry | DA3-GIANT-1.1, `dino_salad.ckpt` |
| [SAM 3.1](https://github.com/facebookresearch/sam3) | Anchor segmentation and tracking | `sam3.1_multiplex.pt` |
| [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B) | Anchor proposals and captions (construction only) | Qwen3.5-9B |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Appearance features | ViT-S/14 |
| [LPIPS](https://github.com/richzhang/PerceptualSimilarity) | Perceptual similarity | v0.1 AlexNet |
| [MUSIQ](https://github.com/chaofengc/IQA-PyTorch) | Imaging quality | SPAQ |
| [LAION aesthetic](https://github.com/LAION-AI/aesthetic-predictor) | Aesthetic quality | CLIP ViT-L/14 + linear head |
| [AMT](https://github.com/MCG-NKU/AMT) (via [VBench](https://github.com/Vchitect/VBench)) | Motion smoothness | AMT-S |
| [HPSv3](https://github.com/MizzenAI/HPSv3) | Human preference | HPSv3 + Qwen2-VL-7B |
