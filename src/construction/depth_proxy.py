"""Sparse depth fusion and voxel-box queries used by route QA."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar, Iterable, Literal, Sequence
import numpy as np
from scipy.spatial import cKDTree
from src.utils.camera import CameraIntrinsics

_DERIVED_PROXY_GRADE = "derived_proxy"

class _DerivedProxyArtifact:
    geometry_grade: ClassVar[Literal['derived_proxy']] = _DERIVED_PROXY_GRADE
    geometry_independent: ClassVar[bool] = False
    mesh_ground_truth: ClassVar[bool] = False

class DerivedProxyVoxelState(IntEnum):
    UNKNOWN = 0
    KNOWN_FREE = 1
    OCCUPIED = 2

@dataclass(frozen=True)
class DerivedProxyCapabilities(_DerivedProxyArtifact):
    geometry_grade: Literal['derived_proxy'] = _DERIVED_PROXY_GRADE
    geometry_independent: bool = False
    collision_grade: Literal['derived_proxy'] = _DERIVED_PROXY_GRADE
    mesh_validated: bool = False
    unknown_space_policy: Literal['reject', 'diagnostic_only'] = 'diagnostic_only'
    depth_source: str = 'registered_depth'

    def __post_init__(self) -> None:
        if self.geometry_grade != _DERIVED_PROXY_GRADE:
            raise ValueError('proxy geometry must be labelled derived_proxy')
        if self.geometry_independent:
            raise ValueError('derived proxy geometry cannot claim independence')
        if self.collision_grade != _DERIVED_PROXY_GRADE:
            raise ValueError('derived proxy collision grade must be derived_proxy')
        if self.mesh_validated:
            raise ValueError('derived proxy geometry cannot claim mesh validation')
        if self.unknown_space_policy not in {'reject', 'diagnostic_only'}:
            raise ValueError('unknown_space_policy must be reject or diagnostic_only')
        if not self.depth_source:
            raise ValueError('depth_source must be non-empty')

@dataclass(frozen=True)
class DerivedProxyFrame(_DerivedProxyArtifact):
    depth_m: np.ndarray
    camera: CameraIntrinsics
    T_world_camera: np.ndarray
    static_mask: np.ndarray | None = None
    frame_id: str = ''
    depth_source: str = 'registered_depth'

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_m, dtype=np.float32)
        if depth.shape != (self.camera.height, self.camera.width):
            raise ValueError(f'depth_m must have shape ({self.camera.height}, {self.camera.width})')
        if self.static_mask is None:
            static_mask = np.ones(depth.shape, dtype=bool)
        else:
            static_mask = np.asarray(self.static_mask, dtype=bool)
            if static_mask.shape != depth.shape:
                raise ValueError('static_mask must have the same shape as depth_m')
        pose = np.asarray(self.T_world_camera, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError('T_world_camera must have shape (4, 4)')
        if not np.isfinite(pose).all():
            raise ValueError('T_world_camera contains non-finite values')
        if not self.depth_source:
            raise ValueError('depth_source must be non-empty')
        depth = depth.copy()
        static_mask = static_mask.copy()
        pose = pose.copy()
        depth.setflags(write=False)
        static_mask.setflags(write=False)
        pose.setflags(write=False)
        object.__setattr__(self, 'depth_m', depth)
        object.__setattr__(self, 'static_mask', static_mask)
        object.__setattr__(self, 'T_world_camera', pose)
        object.__setattr__(self, 'frame_id', str(self.frame_id))

    @property
    def valid_static_depth_mask(self) -> np.ndarray:
        return np.asarray(self.static_mask, dtype=bool) & np.isfinite(self.depth_m) & (self.depth_m > 0.0)

@dataclass(frozen=True)
class DerivedProxyFusionSettings(_DerivedProxyArtifact):
    voxel_size_m: float
    grid_origin_m: np.ndarray = field(default_factory=lambda : np.zeros(3, dtype=np.float64))
    surface_band_m: float | None = None
    pixel_stride: int = 1
    maximum_samples_per_frame: int | None = 50000
    maximum_free_ray_samples: int | None = None
    camera_center_free_radius_m: float = 0.0

    def __post_init__(self) -> None:
        voxel_size = float(self.voxel_size_m)
        if not np.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError('voxel_size_m must be a positive finite value')
        origin = np.asarray(self.grid_origin_m, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError('grid_origin_m must be a finite 3-vector')
        if isinstance(self.pixel_stride, (bool, np.bool_)) or not isinstance(self.pixel_stride, (int, np.integer)) or int(self.pixel_stride) <= 0:
            raise ValueError('pixel_stride must be a positive integer')
        maximum = self.maximum_samples_per_frame
        if maximum is not None and (isinstance(maximum, (bool, np.bool_)) or not isinstance(maximum, (int, np.integer)) or int(maximum) <= 0):
            raise ValueError('maximum_samples_per_frame must be positive or None')
        maximum_ray_samples = self.maximum_free_ray_samples
        if maximum_ray_samples is not None and (isinstance(maximum_ray_samples, (bool, np.bool_)) or not isinstance(maximum_ray_samples, (int, np.integer)) or int(maximum_ray_samples) <= 0):
            raise ValueError('maximum_free_ray_samples must be positive or None')
        band = 0.5 * voxel_size if self.surface_band_m is None else float(self.surface_band_m)
        if not np.isfinite(band) or band < 0.0:
            raise ValueError('surface_band_m must be finite and non-negative')
        center_radius = float(self.camera_center_free_radius_m)
        if not np.isfinite(center_radius) or center_radius < 0.0:
            raise ValueError('camera_center_free_radius_m must be finite and non-negative')
        origin = origin.copy()
        origin.setflags(write=False)
        object.__setattr__(self, 'voxel_size_m', voxel_size)
        object.__setattr__(self, 'grid_origin_m', origin)
        object.__setattr__(self, 'surface_band_m', band)
        object.__setattr__(self, 'pixel_stride', int(self.pixel_stride))
        object.__setattr__(self, 'maximum_samples_per_frame', None if maximum is None else int(maximum))
        object.__setattr__(self, 'maximum_free_ray_samples', None if maximum_ray_samples is None else int(maximum_ray_samples))
        object.__setattr__(self, 'camera_center_free_radius_m', center_radius)

def _readonly_array(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array

def _validate_voxels(value: np.ndarray, name: str) -> np.ndarray:
    voxels = np.asarray(value, dtype=np.int64)
    if voxels.size == 0:
        return _readonly_array(np.empty((0, 3), dtype=np.int64), np.int64)
    if voxels.ndim != 2 or voxels.shape[1] != 3:
        raise ValueError(f'{name} must have shape (N, 3)')
    return _readonly_array(np.unique(voxels, axis=0), np.int64)

@dataclass(frozen=True)
class DerivedProxyGeometry(_DerivedProxyArtifact):
    voxel_size_m: float
    grid_origin_m: np.ndarray
    known_free_voxels: np.ndarray
    occupied_voxels: np.ndarray
    surface_points_world: np.ndarray
    fusion_frame_ids: tuple[str, ...]
    capabilities: DerivedProxyCapabilities = field(default_factory=DerivedProxyCapabilities)
    depth_source: str = 'registered_depth'
    _known_free_set: frozenset[tuple[int, int, int]] = field(init=False, repr=False, compare=False)
    _occupied_set: frozenset[tuple[int, int, int]] = field(init=False, repr=False, compare=False)
    _occupied_center_tree: cKDTree | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        voxel_size = float(self.voxel_size_m)
        if not np.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError('voxel_size_m must be a positive finite value')
        origin = np.asarray(self.grid_origin_m, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError('grid_origin_m must be a finite 3-vector')
        known_free = _validate_voxels(self.known_free_voxels, 'known_free_voxels')
        occupied = _validate_voxels(self.occupied_voxels, 'occupied_voxels')
        free_set = frozenset(map(tuple, known_free.tolist()))
        occupied_set = frozenset(map(tuple, occupied.tolist()))
        if free_set & occupied_set:
            raise ValueError('known-free and occupied voxel sets must not overlap')
        surface = np.asarray(self.surface_points_world, dtype=np.float32)
        if surface.ndim != 2 or surface.shape[1] != 3:
            raise ValueError('surface_points_world must have shape (N, 3)')
        if len(surface) == 0 or not np.isfinite(surface).all():
            raise ValueError('surface_points_world must contain at least one finite point')
        ids = tuple((str(frame_id) for frame_id in self.fusion_frame_ids))
        if len(ids) != len(set(ids)):
            raise ValueError('fusion_frame_ids must be unique')
        if not self.depth_source:
            raise ValueError('depth_source must be non-empty')
        if self.capabilities.geometry_grade != _DERIVED_PROXY_GRADE:
            raise ValueError('capabilities must explicitly identify derived_proxy')
        if self.capabilities.geometry_independent or self.capabilities.mesh_validated:
            raise ValueError('proxy geometry must not claim independent mesh evidence')
        origin = _readonly_array(origin, np.float64)
        surface = _readonly_array(surface, np.float32)
        object.__setattr__(self, 'voxel_size_m', voxel_size)
        object.__setattr__(self, 'grid_origin_m', origin)
        object.__setattr__(self, 'known_free_voxels', known_free)
        object.__setattr__(self, 'occupied_voxels', occupied)
        object.__setattr__(self, 'surface_points_world', surface)
        object.__setattr__(self, 'fusion_frame_ids', ids)
        object.__setattr__(self, '_known_free_set', free_set)
        object.__setattr__(self, '_occupied_set', occupied_set)
        centers = origin + (occupied.astype(np.float64) + 0.5) * voxel_size
        object.__setattr__(self, '_occupied_center_tree', None if len(centers) == 0 else cKDTree(centers))

    @property
    def fusion_frame_count(self) -> int:
        return len(self.fusion_frame_ids)

    def voxel_indices(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError('points_world must have shape (..., 3)')
        if not np.isfinite(points).all():
            raise ValueError('points_world contains non-finite values')
        return np.floor((points - self.grid_origin_m) / self.voxel_size_m).astype(np.int64)

    def voxel_states(self, points_world: np.ndarray) -> np.ndarray:
        indices = self.voxel_indices(points_world)
        flat = indices.reshape(-1, 3)
        states = np.full(len(flat), DerivedProxyVoxelState.UNKNOWN, dtype=np.uint8)
        for (index, voxel) in enumerate(flat):
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
            if key in self._occupied_set:
                states[index] = DerivedProxyVoxelState.OCCUPIED
            elif key in self._known_free_set:
                states[index] = DerivedProxyVoxelState.KNOWN_FREE
        return states.reshape(indices.shape[:-1])

    def point_clearance(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError('points_world must have shape (..., 3)')
        if not np.isfinite(points).all():
            raise ValueError('points_world contains non-finite values')
        original_shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        if len(self.occupied_voxels) == 0:
            return np.full(original_shape, np.inf, dtype=np.float32)
        centers = self.grid_origin_m + (self.occupied_voxels.astype(np.float64) + 0.5) * self.voxel_size_m
        half_size = 0.5 * self.voxel_size_m
        output = np.empty(len(flat), dtype=np.float64)
        assert self._occupied_center_tree is not None
        neighbour_count = min(16, len(centers))
        for start in range(0, len(flat), 4096):
            stop = min(start + 4096, len(flat))
            (_, nearest) = self._occupied_center_tree.query(flat[start:stop], k=neighbour_count, workers=1)
            nearest = np.asarray(nearest, dtype=np.int64)
            if neighbour_count == 1:
                nearest = nearest[:, None]
            nearby_centers = centers[nearest]
            offset = np.abs(flat[start:stop, None, :] - nearby_centers)
            box_gap = np.maximum(offset - half_size, 0.0)
            distances = np.linalg.norm(box_gap, axis=-1)
            output[start:stop] = np.min(distances, axis=1)
        return output.reshape(original_shape).astype(np.float32)

def build_derived_proxy_geometry(fusion_frames: Sequence[DerivedProxyFrame], settings: DerivedProxyFusionSettings) -> DerivedProxyGeometry:
    frames = tuple(fusion_frames)
    if not frames:
        raise ValueError('at least one fusion frame is required')
    if not all((isinstance(frame, DerivedProxyFrame) for frame in frames)):
        raise TypeError('fusion_frames must contain DerivedProxyFrame values')
    known_free: set[tuple[int, int, int]] = set()
    occupied: set[tuple[int, int, int]] = set()
    surface_chunks: list[np.ndarray] = []
    frame_ids: list[str] = []
    carve_step_m = settings.voxel_size_m
    for (frame_index, frame) in enumerate(frames):
        frame_id = frame.frame_id or f'fusion_frame_{frame_index}'
        if frame_id in frame_ids:
            raise ValueError('fusion frame IDs must be unique when provided')
        frame_ids.append(frame_id)
        (points_world, camera_center) = _static_world_points(frame, settings)
        if settings.camera_center_free_radius_m > 0.0:
            valid_depth = frame.depth_m[frame.valid_static_depth_mask]
            if len(valid_depth) and float(np.min(valid_depth)) >= settings.camera_center_free_radius_m + settings.surface_band_m:
                _carve_camera_center_free_volume(known_free, camera_center, radius_m=settings.camera_center_free_radius_m, grid_origin_m=settings.grid_origin_m, voxel_size_m=settings.voxel_size_m)
        if len(points_world) == 0:
            continue
        surface_chunks.append(points_world.astype(np.float32))
        surface_voxels = _point_voxels(points_world, settings.grid_origin_m, settings.voxel_size_m)
        occupied.update(map(tuple, surface_voxels.tolist()))
        for point in points_world:
            _carve_free_ray(known_free, camera_center, point, grid_origin_m=settings.grid_origin_m, voxel_size_m=settings.voxel_size_m, surface_band_m=float(settings.surface_band_m), step_m=carve_step_m, maximum_samples=settings.maximum_free_ray_samples)
    if not surface_chunks:
        raise ValueError('fusion frames contain no valid static depth samples')
    known_free.difference_update(occupied)
    return DerivedProxyGeometry(voxel_size_m=settings.voxel_size_m, grid_origin_m=settings.grid_origin_m, known_free_voxels=_voxel_array(known_free), occupied_voxels=_voxel_array(occupied), surface_points_world=np.concatenate(surface_chunks, axis=0), fusion_frame_ids=tuple(frame_ids))

def _static_world_points(frame: DerivedProxyFrame, settings: DerivedProxyFusionSettings) -> tuple[np.ndarray, np.ndarray]:
    mask = frame.valid_static_depth_mask
    if settings.pixel_stride > 1:
        sampled = np.zeros_like(mask)
        sampled[::settings.pixel_stride, ::settings.pixel_stride] = True
        mask = mask & sampled
    (rows, columns) = np.nonzero(mask)
    if settings.maximum_samples_per_frame is not None and len(rows) > settings.maximum_samples_per_frame:
        selected = np.linspace(0, len(rows) - 1, settings.maximum_samples_per_frame, dtype=np.int64)
        rows = rows[selected]
        columns = columns[selected]
    if len(rows) == 0:
        return (np.empty((0, 3), dtype=np.float64), frame.T_world_camera[:3, 3])
    pixels = np.stack([columns.astype(np.float64) + 0.5, rows.astype(np.float64) + 0.5], axis=-1)
    points_camera = frame.camera.normalized_rays(pixels) * frame.depth_m[rows, columns, None]
    points_world = points_camera @ frame.T_world_camera[:3, :3].T + frame.T_world_camera[:3, 3]
    return (points_world, frame.T_world_camera[:3, 3])

def _point_voxels(points_world: np.ndarray, grid_origin_m: np.ndarray, voxel_size_m: float) -> np.ndarray:
    return np.floor((np.asarray(points_world, dtype=np.float64) - grid_origin_m) / voxel_size_m).astype(np.int64)

def _carve_free_ray(known_free: set[tuple[int, int, int]], camera_center: np.ndarray, surface_point: np.ndarray, *, grid_origin_m: np.ndarray, voxel_size_m: float, surface_band_m: float, step_m: float, maximum_samples: int | None=None) -> None:
    delta = np.asarray(surface_point, dtype=np.float64) - camera_center
    length = float(np.linalg.norm(delta))
    free_length = max(0.0, length - surface_band_m)
    if length <= 1e-12 or free_length <= 0.0:
        return
    count = max(1, int(np.ceil(free_length / step_m)))
    if maximum_samples is not None:
        count = min(count, maximum_samples)
    alphas = np.linspace(0.0, free_length / length, count + 1)
    samples = camera_center + alphas[:, None] * delta
    voxels = _point_voxels(samples, grid_origin_m, voxel_size_m)
    known_free.update(map(tuple, voxels.tolist()))

def _carve_camera_center_free_volume(known_free: set[tuple[int, int, int]], camera_center: np.ndarray, *, radius_m: float, grid_origin_m: np.ndarray, voxel_size_m: float) -> None:
    if radius_m <= 0.0:
        return
    center_voxel = _point_voxels(np.asarray(camera_center, dtype=np.float64)[None], grid_origin_m, voxel_size_m)[0]
    radius_voxels = int(np.ceil(radius_m / voxel_size_m))
    offsets = np.arange(-radius_voxels, radius_voxels + 1, dtype=np.int64)
    (xx, yy, zz) = np.meshgrid(offsets, offsets, offsets, indexing='ij')
    candidates = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    centers = grid_origin_m + (center_voxel[None].astype(np.float64) + candidates + 0.5) * voxel_size_m
    keep = np.linalg.norm(centers - camera_center[None], axis=1) <= radius_m + 0.5 * np.sqrt(3.0) * voxel_size_m
    voxels = center_voxel[None] + candidates[keep]
    known_free.update(map(tuple, voxels.tolist()))

def _voxel_array(voxels: Iterable[tuple[int, int, int]]) -> np.ndarray:
    values = list(voxels)
    if not values:
        return np.empty((0, 3), dtype=np.int64)
    return np.unique(np.asarray(values, dtype=np.int64), axis=0)
