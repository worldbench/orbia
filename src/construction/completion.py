"""Final completion surface and event-pixel calculations."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from src.utils.camera import CameraIntrinsics
from src.utils.geometry import MeshRaycaster, RaycastResult
from src.construction.depth_proxy import _DerivedProxyArtifact


def _visible_primitives(render: RaycastResult) -> np.ndarray:
    values = render.primitive_ids[render.valid]
    return np.unique(values.astype(np.uint64, copy=False))


def _sorted_unique_voxels(values: np.ndarray) -> np.ndarray:
    voxels = np.asarray(values, dtype=np.int64)
    if voxels.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if voxels.ndim != 2 or voxels.shape[1] != 3:
        raise ValueError("surface voxels must have shape (N, 3)")
    return np.unique(voxels, axis=0)


_VOXEL_KEY_DTYPE = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])


_SURFACE_VOXEL_HALF_TIE_EPSILON = 5e-5


def _quantize_surface_points(
    points_world_m: np.ndarray,
    voxel_size_m: float,
) -> np.ndarray:
    points = np.asarray(points_world_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("world surface points must have shape (N, 3)")
    return np.floor(
        points / voxel_size_m + 0.5 + _SURFACE_VOXEL_HALF_TIE_EPSILON
    ).astype(np.int64)


def _voxel_keys(values: np.ndarray) -> np.ndarray:
    canonical = np.ascontiguousarray(values, dtype="<i8")
    if canonical.ndim != 2 or canonical.shape[1] != 3:
        raise ValueError("surface voxels must have shape (N, 3)")
    return canonical.view(_VOXEL_KEY_DTYPE).reshape(-1)


def _voxels_from_keys(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.int64)
    return np.ascontiguousarray(values).view("<i8").reshape(-1, 3).copy()


def _voxel_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return _voxels_from_keys(
        np.intersect1d(_voxel_keys(first), _voxel_keys(second), assume_unique=True)
    )


def _crop_camera_rows(
    camera: CameraIntrinsics, row_start: int, row_end: int
) -> CameraIntrinsics:
    """Return a calibrated row crop without resampling or reducing its FOV."""

    if not 0 <= row_start < row_end <= camera.height:
        raise ValueError("invalid camera row crop")
    K = camera.K.copy()
    K[1, 2] -= row_start
    return CameraIntrinsics(
        width=camera.width,
        height=row_end - row_start,
        K=K,
        model=camera.model,
        distortion=camera.distortion,
    )


def _surface_observation(
    camera: CameraIntrinsics,
    pose_world_camera: np.ndarray,
    raycaster: MeshRaycaster,
    voxel_size_m: float,
    *,
    chunk_rows: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Raycast every camera pixel and return diagnostics plus metric surfels."""

    primitive_chunks: list[np.ndarray] = []
    voxel_chunks: list[np.ndarray] = []
    for row_start in range(0, camera.height, chunk_rows):
        row_end = min(row_start + chunk_rows, camera.height)
        crop = _crop_camera_rows(camera, row_start, row_end)
        render = raycaster.render(crop, pose_world_camera)
        primitive_chunks.append(_visible_primitives(render))
        if np.any(render.valid):
            points = raycaster.hit_points_world(crop, pose_world_camera, render)
            hit_points = points[render.valid]
            quantized = _quantize_surface_points(hit_points, voxel_size_m)
            voxel_chunks.append(_sorted_unique_voxels(quantized))
    primitives = (
        np.unique(np.concatenate(primitive_chunks)).astype(np.uint64, copy=False)
        if primitive_chunks
        else np.empty(0, dtype=np.uint64)
    )
    voxels = (
        _sorted_unique_voxels(np.concatenate(voxel_chunks, axis=0))
        if voxel_chunks
        else np.empty((0, 3), dtype=np.int64)
    )
    return primitives, voxels


def _exclude_surface_near_q0(
    candidates: np.ndarray,
    q0_visible: np.ndarray,
    voxel_size_m: float,
    exclusion_radius_m: float,
) -> np.ndarray:
    """Remove candidates whose closed voxel cell is within ``radius`` of q0."""

    candidates = _sorted_unique_voxels(candidates)
    q0_visible = _sorted_unique_voxels(q0_visible)
    if len(candidates) == 0 or len(q0_visible) == 0:
        return candidates
    q0_centers_m = q0_visible.astype(np.float64) * voxel_size_m
    candidate_centers_m = candidates.astype(np.float64) * voxel_size_m
    broad_phase_radius_m = exclusion_radius_m + np.sqrt(3.0) * voxel_size_m
    tree = cKDTree(q0_centers_m)
    neighbor_indices = tree.query_ball_point(
        candidate_centers_m,
        r=broad_phase_radius_m + 1e-12,
        p=2.0,
        workers=1,
    )
    keep = np.ones(len(candidates), dtype=bool)
    tolerance_m = max(1e-12, 8.0 * np.finfo(np.float64).eps * broad_phase_radius_m)
    for candidate_index, neighbors in enumerate(neighbor_indices):
        if not neighbors:
            continue
        center_deltas_m = np.abs(
            q0_centers_m[np.asarray(neighbors, dtype=np.int64)]
            - candidate_centers_m[candidate_index]
        )
        cell_gaps_m = np.maximum(center_deltas_m - voxel_size_m, 0.0)
        minimum_cell_distances_m = np.linalg.norm(cell_gaps_m, axis=1)
        if np.any(minimum_cell_distances_m <= exclusion_radius_m + tolerance_m):
            keep[candidate_index] = False
    return candidates[keep]


def _event_surface_pixel_voxels(
    camera: CameraIntrinsics,
    pose_world_camera: np.ndarray,
    raycaster: MeshRaycaster,
    voxel_size_m: float,
    *,
    chunk_rows: int = 128,
    minimum_depth_m: float | None = None,
    maximum_depth_m: float | None = None,
) -> np.ndarray:
    """Return one world-surface voxel per valid model-crop mesh-hit pixel."""

    if minimum_depth_m is not None:
        minimum_depth_m = float(minimum_depth_m)
        if not np.isfinite(minimum_depth_m) or minimum_depth_m < 0.0:
            raise ValueError("minimum_depth_m must be finite and non-negative")
    if maximum_depth_m is not None:
        maximum_depth_m = float(maximum_depth_m)
        if not np.isfinite(maximum_depth_m) or maximum_depth_m <= 0.0:
            raise ValueError("maximum_depth_m must be finite and positive")
    if (
        minimum_depth_m is not None
        and maximum_depth_m is not None
        and maximum_depth_m <= minimum_depth_m
    ):
        raise ValueError("maximum_depth_m must exceed minimum_depth_m")

    voxel_chunks: list[np.ndarray] = []
    for row_start in range(0, camera.height, chunk_rows):
        row_end = min(row_start + chunk_rows, camera.height)
        crop = _crop_camera_rows(camera, row_start, row_end)
        render = raycaster.render(crop, pose_world_camera)
        valid = np.asarray(render.valid, dtype=bool).copy()
        if minimum_depth_m is not None:
            valid &= render.depth_m >= minimum_depth_m
        if maximum_depth_m is not None:
            valid &= render.depth_m < maximum_depth_m
        if not np.any(valid):
            continue
        points = raycaster.hit_points_world(crop, pose_world_camera, render)
        voxel_chunks.append(
            _quantize_surface_points(points[valid], voxel_size_m)
        )
    return (
        np.concatenate(voxel_chunks, axis=0)
        if voxel_chunks
        else np.empty((0, 3), dtype=np.int64)
    )


def _shared_event_pixel_metrics(
    first_pixel_voxels: np.ndarray,
    revisit_pixel_voxels: np.ndarray,
    q0_visible_voxels: np.ndarray,
    voxel_size_m: float,
    exclusion_radius_m: float,
) -> tuple[np.ndarray, int, int, float, int, int, float]:
    """Measure the exact common q0-unseen surface in both event crops."""

    first_pixels = np.asarray(first_pixel_voxels, dtype=np.int64).reshape(-1, 3)
    revisit_pixels = np.asarray(revisit_pixel_voxels, dtype=np.int64).reshape(-1, 3)
    first_unseen = _exclude_surface_near_q0(
        _sorted_unique_voxels(first_pixels),
        q0_visible_voxels,
        voxel_size_m,
        exclusion_radius_m,
    )
    revisit_unseen = _exclude_surface_near_q0(
        _sorted_unique_voxels(revisit_pixels),
        q0_visible_voxels,
        voxel_size_m,
        exclusion_radius_m,
    )
    shared = _voxel_intersection(first_unseen, revisit_unseen)
    shared_keys = _voxel_keys(shared)
    first_valid = len(first_pixels)
    revisit_valid = len(revisit_pixels)
    first_shared = int(
        np.count_nonzero(np.isin(_voxel_keys(first_pixels), shared_keys))
    )
    revisit_shared = int(
        np.count_nonzero(np.isin(_voxel_keys(revisit_pixels), shared_keys))
    )
    return (
        shared,
        first_shared,
        first_valid,
        float(first_shared / first_valid) if first_valid else 0.0,
        revisit_shared,
        revisit_valid,
        float(revisit_shared / revisit_valid) if revisit_valid else 0.0,
    )


def completion_strength_grade(
    first_fraction: float,
    revisit_fraction: float,
    *,
    contract_valid: bool,
) -> str:
    """Grade novelty without making it a universal template admission gate."""

    if not contract_valid:
        return "invalid"
    weaker = min(float(first_fraction), float(revisit_fraction))
    if weaker >= 0.40:
        return "strong"
    if weaker >= 0.10:
        return "moderate"
    return "weak"


@dataclass(frozen=True)
class SpatialVIDEventRevisitThresholds:
    """Locally metric event-pair gates for SpatialVID proxy geometry."""

    surface_voxel_size: float = 0.05
    q0_exclusion_radius: float = 0.075
    maximum_pose_translation_error: float = 0.15
    maximum_pose_rotation_error_deg: float = 8.0
    minimum_event_shared_unseen_pixel_fraction: float = 0.40
    minimum_event_valid_proxy_hit_pixel_count: int = 64
    minimum_event_depth: float = 0.5
    maximum_event_depth: float = 8.0
    # Reported as diagnostics only.
    minimum_surface_voxel_jaccard: float = 0.50
    minimum_first_event_coverage: float = 0.65
    minimum_revisit_event_coverage: float = 0.65

    def __post_init__(self) -> None:
        for name in (
            "surface_voxel_size",
            "q0_exclusion_radius",
            "maximum_pose_translation_error",
            "maximum_pose_rotation_error_deg",
            "minimum_event_depth",
            "maximum_event_depth",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in (
            "minimum_surface_voxel_jaccard",
            "minimum_first_event_coverage",
            "minimum_revisit_event_coverage",
            "minimum_event_shared_unseen_pixel_fraction",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        minimum_pixels = self.minimum_event_valid_proxy_hit_pixel_count
        if (
            isinstance(minimum_pixels, (bool, np.bool_))
            or not isinstance(minimum_pixels, (int, np.integer))
            or int(minimum_pixels) <= 0
        ):
            raise ValueError(
                "minimum_event_valid_proxy_hit_pixel_count must be a positive integer"
            )
        object.__setattr__(
            self, "minimum_event_valid_proxy_hit_pixel_count", int(minimum_pixels)
        )
        if self.maximum_event_depth <= self.minimum_event_depth:
            raise ValueError("maximum_event_depth must exceed minimum_event_depth")


@dataclass(frozen=True)
class DerivedProxyRenderResult(_DerivedProxyArtifact):
    """Surface-splat render from fused learned-depth samples, not a mesh render."""

    depth_m: np.ndarray
    valid: np.ndarray
    surface_points_world: np.ndarray
    surface_sample_indices: np.ndarray

    @property
    def primitive_ids(self) -> np.ndarray:
        """Mesh-compatible transient IDs for shared completion bookkeeping."""

        invalid = np.iinfo(np.uint32).max
        values = np.full(self.surface_sample_indices.shape, invalid, dtype=np.uint32)
        valid = np.asarray(self.valid, dtype=bool)
        values[valid] = self.surface_sample_indices[valid].astype(np.uint32)
        return values

    @property
    def normals_world(self) -> np.ndarray:
        """Return zero normals; proxy completion uses hit-point voxels only."""

        return np.zeros(self.surface_points_world.shape, dtype=np.float32)


def _validated_pose(T_world_camera: np.ndarray) -> np.ndarray:
    pose = np.asarray(T_world_camera, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("T_world_camera must have shape (4, 4)")
    if not np.isfinite(pose).all():
        raise ValueError("T_world_camera contains non-finite values")
    return pose


class CompletionProxyRaycaster(_DerivedProxyArtifact):
    """The final nearest-sample z-buffer, with its visibility-only constructor."""

    def __init__(self, surface_points_world):
        self.surface_points_world = np.asarray(surface_points_world, dtype=np.float32).reshape(-1, 3)
    def render(
        self,
        camera: CameraIntrinsics,
        T_world_camera: np.ndarray,
    ) -> DerivedProxyRenderResult:
        """Z-buffer fused surface samples into a calibrated target camera."""

        pose = _validated_pose(T_world_camera)
        rotation_world_camera = pose[:3, :3]
        camera_center = pose[:3, 3]
        points_camera = (
            self.surface_points_world.astype(np.float64) - camera_center
        ) @ rotation_world_camera
        in_front = points_camera[:, 2] > 1e-8

        depth = np.zeros((camera.height, camera.width), dtype=np.float32)
        valid = np.zeros((camera.height, camera.width), dtype=bool)
        surface_points = np.full(
            (camera.height, camera.width, 3), np.nan, dtype=np.float32
        )
        sample_indices = np.full(
            (camera.height, camera.width), -1, dtype=np.int64
        )
        if not np.any(in_front):
            return DerivedProxyRenderResult(
                depth, valid, surface_points, sample_indices
            )

        original_indices = np.flatnonzero(in_front)
        points_camera = points_camera[in_front]
        image_points = camera.project_camera_points(points_camera)
        columns = np.floor(image_points[:, 0]).astype(np.int64)
        rows = np.floor(image_points[:, 1]).astype(np.int64)
        inside = (
            (columns >= 0)
            & (columns < camera.width)
            & (rows >= 0)
            & (rows < camera.height)
        )
        if not np.any(inside):
            return DerivedProxyRenderResult(
                depth, valid, surface_points, sample_indices
            )

        columns = columns[inside]
        rows = rows[inside]
        camera_depths = points_camera[inside, 2]
        original_indices = original_indices[inside]
        flat_pixels = rows * camera.width + columns
        # Group pixel IDs first, then select the nearest depth per pixel.
        order = np.lexsort((camera_depths, flat_pixels))
        sorted_pixels = flat_pixels[order]
        first_for_pixel = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
        winners = order[first_for_pixel]
        winner_pixels = flat_pixels[winners]
        winner_rows = rows[winners]
        winner_columns = columns[winners]
        winner_samples = original_indices[winners]

        depth.flat[winner_pixels] = camera_depths[winners].astype(np.float32)
        valid.flat[winner_pixels] = True
        sample_indices.flat[winner_pixels] = winner_samples
        surface_points[winner_rows, winner_columns] = self.surface_points_world[
            winner_samples
        ]
        return DerivedProxyRenderResult(depth, valid, surface_points, sample_indices)

    def hit_points_world(
        self,
        camera: CameraIntrinsics,
        T_world_camera: np.ndarray,
        result: DerivedProxyRenderResult | None = None,
    ) -> np.ndarray:
        """Return rendered derived surface points and NaN where no proxy hit exists."""

        rendered = result or self.render(camera, T_world_camera)
        return rendered.surface_points_world.copy()
