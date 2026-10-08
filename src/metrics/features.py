"""Offline DINOv2, LPIPS and masked pixel comparisons."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('OpenCV is required for image metrics') from exc
    return cv2

def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left.reshape(-1), right.reshape(-1)) / denominator) if denominator > 1e-12 else 0.0

def _masked_values(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    if mask is None:
        return value.reshape(-1, value.shape[-1])
    if mask.shape != value.shape[:2]:
        raise ValueError('metric mask shape does not match image')
    if not np.any(mask):
        return np.empty((0, value.shape[-1]), dtype=np.float32)
    return value[np.asarray(mask, dtype=bool)]

def _ssim_global(left: np.ndarray, right: np.ndarray, mask: np.ndarray | None) -> float | None:
    cv2 = _cv2()
    left_gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY).astype(np.float64)
    right_gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY).astype(np.float64)
    values_left = left_gray[mask] if mask is not None else left_gray.reshape(-1)
    values_right = right_gray[mask] if mask is not None else right_gray.reshape(-1)
    if len(values_left) < 2:
        return None
    mean_left, mean_right = (float(values_left.mean()), float(values_right.mean()))
    variance_left, variance_right = (float(values_left.var()), float(values_right.var()))
    covariance = float(((values_left - mean_left) * (values_right - mean_right)).mean())
    c1, c2 = ((0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2)
    return float((2 * mean_left * mean_right + c1) * (2 * covariance + c2) / ((mean_left ** 2 + mean_right ** 2 + c1) * (variance_left + variance_right + c2)))

def masked_psnr(left: np.ndarray, right: np.ndarray, mask: np.ndarray | None=None, *, identical_value_db: float=100.0) -> float | None:
    """PSNR over the selected RGB pixels, with a finite perfect-match value."""
    if np.asarray(left).shape != np.asarray(right).shape:
        raise ValueError('PSNR images must have identical shape')
    values_left = _masked_values(left, mask)
    values_right = _masked_values(right, mask)
    if not len(values_left):
        return None
    mse = float(np.mean((values_left.astype(np.float64) - values_right.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return float(identical_value_db)
    return float(min(10.0 * np.log10(255.0 ** 2 / mse), identical_value_db))

@dataclass
class ImageComparator:
    backend: str = 'dinov2'
    dino_repo: Path | None = None
    dino_checkpoint: Path | None = None
    lpips_checkpoint: Path | None = None
    device: str = 'cpu'

    def __post_init__(self) -> None:
        self.backend = str(self.backend).casefold()
        self._dino: Any | None = None
        self._torch: Any | None = None
        self._lpips: Any | None = None
        self._descriptor_cache: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
        self._dino_processor: Any | None = None
        if self.backend != 'dinov2':
            raise ValueError('feature_backend must be dinov2')

    def clear_case_cache(self) -> None:
        """Drop frame descriptors while retaining resident model weights."""
        self._descriptor_cache.clear()

    @property
    def provenance(self) -> dict[str, Any]:
        return {'feature_backend': self.backend, 'dino_repo': str(self.dino_repo) if self.dino_repo else None, 'dino_checkpoint': str(self.dino_checkpoint) if self.dino_checkpoint else None, 'lpips_checkpoint': str(self.lpips_checkpoint) if self.lpips_checkpoint else None, 'device': self.device}

    @property
    def official_spatial_dino_unavailable_reason(self) -> str | None:
        if self.dino_repo is None or not self.dino_repo.is_dir():
            return 'cross-view DINO reprojection requires a local DINOv2 repository'
        if self.dino_checkpoint is None or not self.dino_checkpoint.is_file():
            return 'cross-view DINO reprojection requires a local DINOv2 checkpoint'
        return None

    def _load_dino(self) -> None:
        if self._dino is not None:
            return
        if self.dino_checkpoint is None:
            raise RuntimeError('DINO backend requires a local checkpoint/model directory; network download is disabled')
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('dinov2 backend requires torch in the evaluator environment') from exc
        if self.dino_repo is None or not self.dino_repo.is_dir() or (not self.dino_checkpoint.is_file()):
            raise FileNotFoundError('dinov2 requires local dino_repo and dino_checkpoint file')
        model = torch.hub.load(str(self.dino_repo), 'dinov2_vits14', source='local', pretrained=False)
        checkpoint = torch.load(self.dino_checkpoint, map_location='cpu')
        state = checkpoint.get('teacher', checkpoint.get('model', checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        cleaned = {str(key).removeprefix('module.'): value for key, value in state.items()}
        model.load_state_dict(cleaned, strict=True)
        self._torch = torch
        self._dino = model.eval().to(self.device)

    def _dino_descriptor(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._load_dino()
        assert self._torch is not None and self._dino is not None
        cv2 = _cv2()
        resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = self._torch.from_numpy(resized).permute(2, 0, 1).float()[None] / 255.0
        tensor = (tensor - self._torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]) / self._torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        with self._torch.no_grad():
            output = self._dino.forward_features(tensor.to(self.device))
        global_feature = output['x_norm_clstoken'].detach().cpu().numpy()[0].astype(np.float32)
        patches = output['x_norm_patchtokens'].detach().cpu().numpy()[0].astype(np.float32)
        return (global_feature, patches)

    def _dino_descriptors(self, images: list[np.ndarray], *, batch_size: int=16) -> list[tuple[np.ndarray, np.ndarray]]:
        """Batched equivalent of ``_dino_descriptor`` for resident scoring."""
        if not images:
            return []
        self._load_dino()
        assert self._torch is not None and self._dino is not None
        cv2 = _cv2()
        mean = self._torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = self._torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        results: list[tuple[np.ndarray, np.ndarray]] = []
        for start in range(0, len(images), batch_size):
            tensors = [self._torch.from_numpy(cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)).permute(2, 0, 1).float() / 255.0 for image in images[start:start + batch_size]]
            tensor = (self._torch.stack(tensors) - mean) / std
            with self._torch.no_grad():
                output = self._dino.forward_features(tensor.to(self.device))
            globals_ = output['x_norm_clstoken'].detach().cpu().numpy().astype(np.float32)
            patches = output['x_norm_patchtokens'].detach().cpu().numpy().astype(np.float32)
            results.extend(((global_, patch) for global_, patch in zip(globals_, patches)))
        return results

    @staticmethod
    def _masked_crop(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Crop a metric mask's support without treating it as a resize mask."""
        rows, columns = np.nonzero(mask)
        if len(rows) == 0:
            return image[:1, :1]
        y0, y1 = (int(rows.min()), int(rows.max()) + 1)
        x0, x1 = (int(columns.min()), int(columns.max()) + 1)
        return image[y0:y1, x0:x1]

    def _descriptor(self, image: np.ndarray, mask: np.ndarray | None, *, cache_key: Any | None=None) -> tuple[np.ndarray, np.ndarray, str]:
        cached = self._descriptor_cache.get(cache_key) if cache_key is not None else None
        if cached is not None:
            return (*cached, self.backend)
        value = self._dino_descriptor(self._masked_crop(image, mask) if mask is not None else image)
        name = 'dinov2'
        if cache_key is not None:
            self._descriptor_cache[cache_key] = value
        return (*value, name)

    def _load_lpips(self) -> None:
        if self._lpips is not None:
            return
        if self.lpips_checkpoint is None:
            return
        if not self.lpips_checkpoint.is_file():
            raise FileNotFoundError('configured local LPIPS backbone checkpoint is missing')
        try:
            import lpips
            import torch
        except ImportError as exc:
            raise RuntimeError('LPIPS requires the optional lpips package') from exc
        hub_dir = self.lpips_checkpoint.parent / 'torch-home' / 'hub'
        alexnet = hub_dir / 'checkpoints' / 'alexnet-owt-7be5be79.pth'
        if not alexnet.is_file():
            raise FileNotFoundError('local LPIPS AlexNet trunk is missing')
        torch.hub.set_dir(str(hub_dir))
        self._torch = torch
        self._lpips = lpips.LPIPS(net='alex', model_path=str(self.lpips_checkpoint), verbose=False).eval().to(self.device)

    def _lpips_distance(self, left: np.ndarray, right: np.ndarray, mask: np.ndarray | None) -> float | None:
        if self.lpips_checkpoint is None:
            return None
        self._load_lpips()
        assert self._torch is not None and self._lpips is not None
        if mask is not None:
            left, right = (self._masked_crop(left, mask), self._masked_crop(right, mask))
        cv2 = _cv2()
        left = cv2.resize(left, (224, 224), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (224, 224), interpolation=cv2.INTER_AREA)
        first = self._torch.from_numpy(left).permute(2, 0, 1).float()[None].to(self.device) / 127.5 - 1.0
        second = self._torch.from_numpy(right).permute(2, 0, 1).float()[None].to(self.device) / 127.5 - 1.0
        with self._torch.no_grad():
            return float(self._lpips(first, second).detach().cpu().item())

    def _lpips_distances(self, left_images: list[np.ndarray], right_images: list[np.ndarray], masks: list[np.ndarray | None], *, batch_size: int=16) -> list[float | None]:
        if self.lpips_checkpoint is None:
            return [None] * len(left_images)
        self._load_lpips()
        assert self._torch is not None and self._lpips is not None
        cv2 = _cv2()
        left_tensors, right_tensors = ([], [])
        for left, right, mask in zip(left_images, right_images, masks):
            if mask is not None:
                left, right = (self._masked_crop(left, mask), self._masked_crop(right, mask))
            left = cv2.resize(left, (224, 224), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (224, 224), interpolation=cv2.INTER_AREA)
            left_tensors.append(self._torch.from_numpy(left).permute(2, 0, 1).float() / 127.5 - 1.0)
            right_tensors.append(self._torch.from_numpy(right).permute(2, 0, 1).float() / 127.5 - 1.0)
        values: list[float | None] = []
        for start in range(0, len(left_tensors), batch_size):
            first = self._torch.stack(left_tensors[start:start + batch_size]).to(self.device)
            second = self._torch.stack(right_tensors[start:start + batch_size]).to(self.device)
            with self._torch.no_grad():
                output = self._lpips(first, second).detach().cpu().reshape(-1).tolist()
            values.extend((float(value) for value in output))
        return values

    def compare_batch(self, left_images: list[np.ndarray], right_images: list[np.ndarray], *, masks: list[np.ndarray | None] | None=None, batch_size: int=16) -> list[dict[str, float | None | str]]:
        """Compare independent image pairs while batching learned backends."""
        if len(left_images) != len(right_images):
            raise ValueError('image-pair batches must have identical lengths')
        masks = [None] * len(left_images) if masks is None else masks
        if len(masks) != len(left_images):
            raise ValueError('mask batch length does not match image pairs')
        for left, right, mask in zip(left_images, right_images, masks):
            if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
                raise ValueError('images must be RGB arrays with identical shape')
            if mask is not None and np.asarray(mask).shape != left.shape[:2]:
                raise ValueError('metric mask shape does not match images')
        cropped_left = [self._masked_crop(image, mask) if mask is not None else image for image, mask in zip(left_images, masks)]
        cropped_right = [self._masked_crop(image, mask) if mask is not None else image for image, mask in zip(right_images, masks)]
        descriptors = self._dino_descriptors(cropped_left + cropped_right, batch_size=batch_size)
        left_descriptors = descriptors[:len(left_images)]
        right_descriptors = descriptors[len(left_images):]
        lpips_values = self._lpips_distances(left_images, right_images, masks, batch_size=batch_size)
        results: list[dict[str, float | None | str]] = []
        for left, right, mask, left_descriptor, right_descriptor, lpips_value in zip(left_images, right_images, masks, left_descriptors, right_descriptors, lpips_values):
            global_left, patch_left = left_descriptor
            global_right, patch_right = right_descriptor
            values_left, values_right = (_masked_values(left, mask), _masked_values(right, mask))
            results.append({'feature_backend': 'dinov2', 'global_feature_similarity': _safe_cosine(global_left, global_right), 'patch_feature_similarity': float(np.mean([_safe_cosine(a, b) for a, b in zip(patch_left, patch_right)])), 'dino_similarity': float(np.mean([_safe_cosine(a, b) for a, b in zip(patch_left, patch_right)])), 'psnr': masked_psnr(left, right, mask), 'lpips': lpips_value, 'perceptual_l1': float(np.mean(np.abs(values_left - values_right)) / 255.0) if len(values_left) else None, 'ssim': _ssim_global(left, right, mask), 'edge_consistency': None, 'valid_pixel_ratio': float(np.asarray(mask, dtype=bool).mean()) if mask is not None else 1.0})
        return results

    def compare(self, left: np.ndarray, right: np.ndarray, *, mask: np.ndarray | None=None) -> dict[str, float | None | str]:
        if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
            raise ValueError('images must be RGB arrays with identical shape')
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != left.shape[:2]:
                raise ValueError('metric mask shape does not match images')
        coverage = float(mask.mean()) if mask is not None else 1.0
        global_left, patch_left, feature_name = self._descriptor(left, mask)
        global_right, patch_right, _ = self._descriptor(right, mask)
        feature_similarity = _safe_cosine(global_left, global_right)
        patch_similarity = float(np.mean([_safe_cosine(a, b) for a, b in zip(patch_left, patch_right)]))
        values_left, values_right = (_masked_values(left, mask), _masked_values(right, mask))
        perceptual_l1 = float(np.mean(np.abs(values_left - values_right)) / 255.0) if len(values_left) else None
        return {'feature_backend': feature_name, 'global_feature_similarity': feature_similarity, 'patch_feature_similarity': patch_similarity, 'dino_similarity': patch_similarity, 'psnr': masked_psnr(left, right, mask), 'lpips': self._lpips_distance(left, right, mask), 'perceptual_l1': perceptual_l1, 'ssim': _ssim_global(left, right, mask), 'edge_consistency': None, 'valid_pixel_ratio': coverage}

    def local_dino_similarity(self, left: np.ndarray, right: np.ndarray, *, mask: np.ndarray | None=None, radius: int=1) -> float | None:
        """Patch similarity with a small target-neighbourhood tolerance."""
        if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
            raise ValueError('images must be RGB arrays with identical shape')
        if radius < 0:
            raise ValueError('local DINO radius must be non-negative')
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != left.shape[:2]:
                raise ValueError('metric mask shape does not match images')
            if not np.any(mask):
                return None
        _left_global, left_patches, _backend = self._descriptor(left, None, cache_key=('local_dino', id(left)))
        _right_global, right_patches, _backend = self._descriptor(right, None, cache_key=('local_dino', id(right)))
        left_grid = self._patch_grid(left_patches)
        right_grid = self._patch_grid(right_patches)
        if left_grid.shape[:2] != right_grid.shape[:2]:
            raise ValueError('left and right descriptor grids do not match')
        if mask is None:
            selected = np.ones(left_grid.shape[:2], dtype=bool)
        else:
            height, width = mask.shape
            selected = np.zeros(left_grid.shape[:2], dtype=bool)
            for row in range(left_grid.shape[0]):
                y0, y1 = (row * height // left_grid.shape[0], (row + 1) * height // left_grid.shape[0])
                for column in range(left_grid.shape[1]):
                    x0, x1 = (column * width // left_grid.shape[1], (column + 1) * width // left_grid.shape[1])
                    selected[row, column] = bool(np.any(mask[y0:y1, x0:x1]))
        scores: list[float] = []
        for row, column in np.argwhere(selected):
            y0, y1 = (max(0, row - radius), min(right_grid.shape[0], row + radius + 1))
            x0, x1 = (max(0, column - radius), min(right_grid.shape[1], column + radius + 1))
            scores.append(max((_safe_cosine(left_grid[row, column], candidate) for candidate in right_grid[y0:y1, x0:x1].reshape(-1, right_grid.shape[-1]))))
        return float(np.mean(scores)) if scores else None

    def compare_corresponding_pixels(self, source: np.ndarray, target: np.ndarray, *, source_rows: np.ndarray, source_columns: np.ndarray, target_rows: np.ndarray, target_columns: np.ndarray, source_cache_key: Any | None=None, target_cache_key: Any | None=None) -> dict[str, float | None | str]:
        """Compare pixels joined by geometry, rather than same image coordinates."""
        source_rows = np.asarray(source_rows, dtype=np.int64)
        source_columns = np.asarray(source_columns, dtype=np.int64)
        target_rows = np.asarray(target_rows, dtype=np.int64)
        target_columns = np.asarray(target_columns, dtype=np.int64)
        if not len(source_rows) == len(source_columns) == len(target_rows) == len(target_columns):
            raise ValueError('geometry correspondence arrays need equal length')
        if not len(source_rows):
            raise ValueError('geometry correspondence is empty')
        source_rgb = source[source_rows, source_columns].astype(np.float32)
        target_rgb = target[target_rows, target_columns].astype(np.float32)
        photometric_l1 = float(np.mean(np.abs(source_rgb - target_rgb)) / 255.0)
        cv2 = _cv2()
        source_edges = cv2.Canny(cv2.cvtColor(source, cv2.COLOR_RGB2GRAY), 80, 160) > 0
        target_edges = cv2.Canny(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY), 80, 160) > 0
        left_edge = source_edges[source_rows, source_columns]
        right_edge = target_edges[target_rows, target_columns]
        denominator = int(left_edge.sum() + right_edge.sum())
        edge_alignment = float(2 * np.logical_and(left_edge, right_edge).sum() / denominator) if denominator else None
        _source_global, source_patches, feature_name = self._descriptor(source, None, cache_key=source_cache_key)
        _target_global, target_patches, _ = self._descriptor(target, None, cache_key=target_cache_key)
        source_grid = self._patch_grid(source_patches)
        target_grid = self._patch_grid(target_patches)
        source_patch_rows = np.minimum(source_rows * source_grid.shape[0] // source.shape[0], source_grid.shape[0] - 1)
        source_patch_columns = np.minimum(source_columns * source_grid.shape[1] // source.shape[1], source_grid.shape[1] - 1)
        target_patch_rows = np.minimum(target_rows * target_grid.shape[0] // target.shape[0], target_grid.shape[0] - 1)
        target_patch_columns = np.minimum(target_columns * target_grid.shape[1] // target.shape[1], target_grid.shape[1] - 1)
        patch_pairs = np.unique(np.stack([source_patch_rows * source_grid.shape[1] + source_patch_columns, target_patch_rows * target_grid.shape[1] + target_patch_columns], axis=1), axis=0)
        source_flat, target_flat = (source_grid.reshape(-1, source_grid.shape[-1]), target_grid.reshape(-1, target_grid.shape[-1]))
        feature_similarity = float(np.mean([_safe_cosine(source_flat[left], target_flat[right]) for left, right in patch_pairs]))
        source_supported = np.unique(patch_pairs[:, 0])
        target_supported = np.unique(patch_pairs[:, 1])
        return {'feature_backend': feature_name, 'photometric_l1': photometric_l1, 'feature_reprojection_similarity': feature_similarity, 'edge_alignment': edge_alignment, 'unique_patch_pair_count': int(len(patch_pairs)), 'source_supported_patch_count': int(len(source_supported)), 'target_supported_patch_count': int(len(target_supported)), 'source_total_patch_count': int(len(source_flat)), 'target_total_patch_count': int(len(target_flat)), 'source_patch_coverage': float(len(source_supported) / max(len(source_flat), 1)), 'target_patch_coverage': float(len(target_supported) / max(len(target_flat), 1))}

    def compare_corresponding_lpips_crops(self, source: np.ndarray, target: np.ndarray, *, source_rows: np.ndarray, source_columns: np.ndarray, target_rows: np.ndarray, target_columns: np.ndarray, max_crops: int=4) -> dict[str, float | int | None]:
        """LPIPS on high-coverage target-view crops of a z-buffered RGB warp."""
        if self.lpips_checkpoint is None or not len(target_rows):
            return {'lpips': None, 'lpips_crop_count': 0}
        warped, valid = (np.asarray(target).copy(), np.zeros(target.shape[:2], dtype=bool))
        warped[target_rows, target_columns] = source[source_rows, source_columns]
        valid[target_rows, target_columns] = True
        height, width = target.shape[:2]
        crop = min(96, height, width)
        if crop < 32:
            return {'lpips': None, 'lpips_crop_count': 0}
        values: list[float] = []
        for row in np.linspace(0, height - crop, 3, dtype=np.int64):
            for column in np.linspace(0, width - crop, 3, dtype=np.int64):
                if float(valid[row:row + crop, column:column + crop].mean()) < 0.9:
                    continue
                value = self._lpips_distance(warped[row:row + crop, column:column + crop], target[row:row + crop, column:column + crop], None)
                if value is not None:
                    values.append(value)
                if len(values) >= max_crops:
                    break
            if len(values) >= max_crops:
                break
        return {'lpips': float(np.median(values)) if values else None, 'lpips_crop_count': len(values)}

    @staticmethod
    def _patch_grid(value: np.ndarray) -> np.ndarray:
        """Normalise handcrafted and DINO patch descriptors to HxWxC."""
        patches = np.asarray(value)
        if patches.ndim == 3:
            return patches
        if patches.ndim != 2:
            raise ValueError('patch descriptor must be HxWxC or NxC')
        side = int(round(np.sqrt(len(patches))))
        if side * side != len(patches):
            raise ValueError('DINO patch token count is not square')
        return patches.reshape(side, side, patches.shape[-1])
