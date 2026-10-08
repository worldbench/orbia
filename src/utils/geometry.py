"""CPU mesh queries backed by Open3D's tensor raycasting API."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from src.utils.camera import CameraIntrinsics


@dataclass(frozen=True)
class RaycastResult:
    """Per-pixel first-hit mesh information in a requested camera."""

    depth_m: np.ndarray
    primitive_ids: np.ndarray
    normals_world: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class PathCollisionReport:
    """Deterministic clearance and segment-intersection validation result."""

    valid: bool
    minimum_clearance_m: float
    required_clearance_m: float
    sampled_position_count: int
    intersected_segment_indices: tuple[int, ...]


@dataclass(frozen=True)
class SampledPathCollisionReport(PathCollisionReport):
    """Camera-to-surface clearance at sampled poses and named events."""

    validation_contract: str
    maximum_sample_spacing_m: float
    desired_continuous_clearance_m: float
    required_sampled_clearance_m: float


def _import_open3d():
    """Import Open3D with a useful error for minimal Conda containers."""

    try:
        import open3d as o3d
    except (ImportError, OSError) as exc:  # pragma: no cover - environment-specific
        hint = ""
        if "libGL.so" in str(exc):
            prefix = Path(os.environ.get("CONDA_PREFIX", ""))
            hint = (
                " Set LD_LIBRARY_PATH to the active Conda environment's lib "
                f"directory ({prefix / 'lib'})."
            )
        raise RuntimeError(
            "Open3D tensor raycasting is required; install open3d."
            + hint
        ) from exc
    return o3d


def _arc_length_sample_positions(
    positions: np.ndarray,
    *,
    uniform_sample_count: int,
    key_position_indices: list[int],
) -> tuple[np.ndarray, float]:
    """Sample the actual piecewise-linear camera path by arc length."""

    deltas = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(
        [np.asarray([0.0], dtype=np.float64), np.cumsum(segment_lengths)]
    )
    total_length = float(cumulative[-1])
    uniform_distances = np.linspace(
        0.0, total_length, int(uniform_sample_count), dtype=np.float64
    )
    # Key nodes are additive evidence.  Keep duplicate endpoint/key samples so
    # the persisted count states exactly which policy was used.
    sample_distances = np.sort(
        np.concatenate(
            [
                uniform_distances,
                cumulative[np.asarray(key_position_indices, dtype=np.int64)],
            ]
        )
    )
    maximum_gap = float(np.max(np.diff(sample_distances), initial=0.0))
    samples = np.empty((len(sample_distances), 3), dtype=np.float64)
    for sample_index, distance in enumerate(sample_distances):
        if total_length <= 1e-12:
            samples[sample_index] = positions[0]
            continue
        segment_index = min(
            int(np.searchsorted(cumulative, distance, side="right") - 1),
            len(segment_lengths) - 1,
        )
        while (
            segment_index < len(segment_lengths) - 1
            and segment_lengths[segment_index] <= 1e-12
        ):
            segment_index += 1
        length = float(segment_lengths[segment_index])
        alpha = (
            0.0
            if length <= 1e-12
            else float((distance - cumulative[segment_index]) / length)
        )
        samples[sample_index] = (
            positions[segment_index]
            + np.clip(alpha, 0.0, 1.0) * deltas[segment_index]
        )
    return samples, maximum_gap


def camera_rays_world(
    camera: CameraIntrinsics,
    T_world_camera: np.ndarray,
) -> np.ndarray:
    """Return ``HxWx6`` world rays whose hit parameter is camera-z depth."""

    pose = np.asarray(T_world_camera, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("T_world_camera must have shape (4, 4)")
    if not np.isfinite(pose).all():
        raise ValueError("T_world_camera contains non-finite values")
    return _camera_rays_world_rows(camera, pose, 0, camera.height)


def _camera_rays_world_rows(
    camera: CameraIntrinsics,
    T_world_camera: np.ndarray,
    row_start: int,
    row_stop: int,
) -> np.ndarray:
    """Return a bounded row slab of camera rays."""

    pose = np.asarray(T_world_camera, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("T_world_camera must have shape (4, 4)")
    if not np.isfinite(pose).all():
        raise ValueError("T_world_camera contains non-finite values")
    if not 0 <= row_start < row_stop <= camera.height:
        raise ValueError("camera ray row bounds are invalid")
    u, v = np.meshgrid(
        np.arange(camera.width, dtype=np.float64) + 0.5,
        np.arange(row_start, row_stop, dtype=np.float64) + 0.5,
    )
    pixels = np.stack([u, v], axis=-1)
    directions_camera = camera.normalized_rays(pixels)
    directions_world = directions_camera @ pose[:3, :3].T
    origins = np.broadcast_to(pose[:3, 3], directions_world.shape)
    return np.concatenate([origins, directions_world], axis=-1).astype(np.float32)


class MeshRaycaster:
    """Read-only triangle-mesh queries on CPU, with no headless GL renderer."""

    invalid_primitive_id = np.iinfo(np.uint32).max

    def __init__(self, mesh_path: Path | str) -> None:
        self.mesh_path = Path(mesh_path).resolve()
        if not self.mesh_path.is_file():
            raise FileNotFoundError(self.mesh_path)
        o3d = _import_open3d()
        legacy = o3d.io.read_triangle_mesh(str(self.mesh_path), enable_post_processing=False)
        if legacy.is_empty() or not legacy.has_triangles():
            raise ValueError(f"mesh has no triangles: {self.mesh_path}")
        self.vertices = np.asarray(legacy.vertices, dtype=np.float64).copy()
        self.triangles = np.asarray(legacy.triangles, dtype=np.int64).copy()
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
        self._o3d = o3d
        self._scene = o3d.t.geometry.RaycastingScene()
        self.geometry_id = int(self._scene.add_triangles(tensor_mesh))

    @classmethod
    def from_arrays(
        cls,
        vertices: np.ndarray,
        triangles: np.ndarray,
    ) -> "MeshRaycaster":
        """Construct an in-memory raycaster, primarily for synthetic tests."""

        vertices = np.asarray(vertices, dtype=np.float32)
        triangles = np.asarray(triangles, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (M, 3)")
        if len(vertices) == 0 or len(triangles) == 0:
            raise ValueError("synthetic mesh must have vertices and triangles")
        o3d = _import_open3d()
        obj = cls.__new__(cls)
        obj.mesh_path = Path("<in-memory>")
        obj.vertices = vertices.astype(np.float64)
        obj.triangles = triangles.copy()
        tensor_mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.Int64),
        )
        obj._o3d = o3d
        obj._scene = o3d.t.geometry.RaycastingScene()
        obj.geometry_id = int(obj._scene.add_triangles(tensor_mesh))
        return obj

    def render(
        self,
        camera: CameraIntrinsics,
        T_world_camera: np.ndarray,
        *,
        maximum_rays_per_chunk: int = 524_288,
    ) -> RaycastResult:
        """Raycast metric depth, triangle IDs, and world normals in row chunks."""

        if (
            isinstance(maximum_rays_per_chunk, (bool, np.bool_))
            or not isinstance(maximum_rays_per_chunk, (int, np.integer))
            or int(maximum_rays_per_chunk) <= 0
        ):
            raise ValueError("maximum_rays_per_chunk must be a positive integer")
        rows_per_chunk = max(1, int(maximum_rays_per_chunk) // camera.width)
        depth = np.empty((camera.height, camera.width), dtype=np.float32)
        primitive_ids = np.empty(
            (camera.height, camera.width), dtype=np.uint32
        )
        normals = np.empty((camera.height, camera.width, 3), dtype=np.float32)
        valid = np.empty((camera.height, camera.width), dtype=bool)
        for row_start in range(0, camera.height, rows_per_chunk):
            row_stop = min(camera.height, row_start + rows_per_chunk)
            rays = _camera_rays_world_rows(
                camera, T_world_camera, row_start, row_stop
            )
            answers = self._scene.cast_rays(self._o3d.core.Tensor(rays))
            slab_depth = answers["t_hit"].numpy().astype(np.float32, copy=False)
            slab_primitives = answers["primitive_ids"].numpy().astype(
                np.uint32, copy=False
            )
            slab_normals = answers["primitive_normals"].numpy().astype(
                np.float32, copy=False
            )
            slab_valid = np.isfinite(slab_depth) & (
                slab_primitives != self.invalid_primitive_id
            )
            depth[row_start:row_stop] = np.where(
                slab_valid, slab_depth, 0.0
            )
            primitive_ids[row_start:row_stop] = slab_primitives
            normals[row_start:row_stop] = np.where(
                slab_valid[..., None], slab_normals, 0.0
            )
            valid[row_start:row_stop] = slab_valid
        return RaycastResult(depth, primitive_ids, normals, valid)

    def hit_points_world(
        self,
        camera: CameraIntrinsics,
        T_world_camera: np.ndarray,
        result: RaycastResult | None = None,
    ) -> np.ndarray:
        """Return ``HxWx3`` first-hit positions, NaN at pixels without a hit."""

        result = result or self.render(camera, T_world_camera)
        rays = camera_rays_world(camera, T_world_camera)
        points = rays[..., :3] + result.depth_m[..., None] * rays[..., 3:]
        return np.where(result.valid[..., None], points, np.nan)

    def point_clearance(self, points_world: np.ndarray) -> np.ndarray:
        """Return unsigned distance from each point to the closest mesh surface."""

        points = np.asarray(points_world, dtype=np.float32)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("points_world must have shape (..., 3)")
        original_shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        if not np.isfinite(flat).all():
            raise ValueError("points_world contains non-finite values")
        distances = self._scene.compute_distance(self._o3d.core.Tensor(flat)).numpy()
        return distances.reshape(original_shape).astype(np.float32)

    def validate_camera_path(
        self,
        camera_positions_world: np.ndarray,
        minimum_clearance_m: float,
        *,
        sample_spacing_m: float = 0.05,
    ) -> PathCollisionReport:
        """Validate clearance and reject paths whose segments cross a mesh."""

        positions = np.asarray(camera_positions_world, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
            raise ValueError("camera_positions_world must have shape (N>=2, 3)")
        if not np.isfinite(positions).all():
            raise ValueError("camera path contains non-finite positions")
        minimum_clearance_m = float(minimum_clearance_m)
        sample_spacing_m = float(sample_spacing_m)
        if minimum_clearance_m <= 0 or sample_spacing_m <= 0:
            raise ValueError("clearance and sample spacing must be positive")

        samples: list[np.ndarray] = [positions[0]]
        segment_rays: list[np.ndarray] = []
        segment_lengths: list[float] = []
        for start, end in zip(positions[:-1], positions[1:]):
            delta = end - start
            length = float(np.linalg.norm(delta))
            if length <= 1e-9:
                continue
            count = max(1, int(np.ceil(length / sample_spacing_m)))
            for alpha in np.linspace(0.0, 1.0, count + 1)[1:]:
                samples.append(start + alpha * delta)
            segment_rays.append(np.concatenate([start, delta / length]))
            segment_lengths.append(length)

        sample_array = np.asarray(samples, dtype=np.float32)
        clearances = self.point_clearance(sample_array)
        minimum = float(np.min(clearances))
        intersections: list[int] = []
        if segment_rays:
            answers = self._scene.cast_rays(
                self._o3d.core.Tensor(np.asarray(segment_rays, dtype=np.float32))
            )
            hits = answers["t_hit"].numpy()
            intersections = [
                index
                for index, (hit, length) in enumerate(zip(hits, segment_lengths))
                if np.isfinite(hit) and float(hit) <= length + 1e-5
            ]
        valid = minimum >= minimum_clearance_m and not intersections
        return PathCollisionReport(
            valid=valid,
            minimum_clearance_m=minimum,
            required_clearance_m=minimum_clearance_m,
            sampled_position_count=len(sample_array),
            intersected_segment_indices=tuple(intersections),
        )

    def validate_camera_path_sampled(
        self,
        camera_positions_world: np.ndarray,
        desired_continuous_clearance_m: float,
        *,
        uniform_sample_count: int = 96,
        key_position_indices: Iterable[int] = (),
    ) -> SampledPathCollisionReport:
        """Validate a path from a bounded sample slate plus segment crossings."""

        positions = np.asarray(camera_positions_world, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
            raise ValueError("camera_positions_world must have shape (N>=2, 3)")
        if not np.isfinite(positions).all():
            raise ValueError("camera path contains non-finite positions")
        desired = float(desired_continuous_clearance_m)
        if not np.isfinite(desired) or desired <= 0.0:
            raise ValueError(
                "desired_continuous_clearance_m must be finite and positive"
            )
        if (
            isinstance(uniform_sample_count, (bool, np.bool_))
            or not isinstance(uniform_sample_count, (int, np.integer))
            or not 64 <= int(uniform_sample_count) <= 96
        ):
            raise ValueError("uniform_sample_count must be an integer in [64, 96]")
        key_indices: list[int] = []
        for raw_index in key_position_indices:
            if (
                isinstance(raw_index, (bool, np.bool_))
                or not isinstance(raw_index, (int, np.integer))
                or not 0 <= int(raw_index) < len(positions)
            ):
                raise ValueError("key_position_indices contains an invalid index")
            key_indices.append(int(raw_index))

        samples, maximum_gap = _arc_length_sample_positions(
            positions,
            uniform_sample_count=int(uniform_sample_count),
            key_position_indices=key_indices,
        )

        clearances = self.point_clearance(samples.astype(np.float32))
        sampled_minimum = float(np.min(clearances))
        required_sampled = float(desired + 0.5 * maximum_gap)

        deltas = np.diff(positions, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        segment_rays: list[np.ndarray] = []
        nonzero_lengths: list[float] = []
        source_segment_indices: list[int] = []
        for segment_index, (start, delta, length) in enumerate(
            zip(positions[:-1], deltas, segment_lengths)
        ):
            length = float(length)
            if length <= 1e-9:
                continue
            segment_rays.append(np.concatenate([start, delta / length]))
            nonzero_lengths.append(length)
            source_segment_indices.append(segment_index)
        intersections: list[int] = []
        if segment_rays:
            answers = self._scene.cast_rays(
                self._o3d.core.Tensor(np.asarray(segment_rays, dtype=np.float32))
            )
            hits = answers["t_hit"].numpy()
            intersections = [
                source_index
                for source_index, hit, length in zip(
                    source_segment_indices, hits, nonzero_lengths
                )
                if np.isfinite(hit) and float(hit) <= length + 1e-5
            ]
        return SampledPathCollisionReport(
            valid=bool(sampled_minimum >= required_sampled and not intersections),
            minimum_clearance_m=sampled_minimum,
            required_clearance_m=required_sampled,
            sampled_position_count=len(samples),
            intersected_segment_indices=tuple(intersections),
            validation_contract="sampled_mesh_clearance",
            maximum_sample_spacing_m=maximum_gap,
            desired_continuous_clearance_m=desired,
            required_sampled_clearance_m=required_sampled,
        )

    def validate_camera_path_coarse_sampled(
        self,
        camera_positions_world: np.ndarray,
        minimum_clearance_m: float,
        *,
        uniform_sample_count: int = 32,
        key_position_indices: Iterable[int] = (),
    ) -> SampledPathCollisionReport:
        """Point-only coarse metric-clearance check."""

        positions = np.asarray(camera_positions_world, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
            raise ValueError("camera_positions_world must have shape (N>=2, 3)")
        if not np.isfinite(positions).all():
            raise ValueError("camera path contains non-finite positions")
        required = float(minimum_clearance_m)
        if not np.isfinite(required) or required <= 0.0:
            raise ValueError("minimum_clearance_m must be finite and positive")
        if (
            isinstance(uniform_sample_count, (bool, np.bool_))
            or not isinstance(uniform_sample_count, (int, np.integer))
            or not 16 <= int(uniform_sample_count) <= 64
        ):
            raise ValueError("uniform_sample_count must be an integer in [16, 64]")
        key_indices: list[int] = []
        for raw_index in key_position_indices:
            if (
                isinstance(raw_index, (bool, np.bool_))
                or not isinstance(raw_index, (int, np.integer))
                or not 0 <= int(raw_index) < len(positions)
            ):
                raise ValueError("key_position_indices contains an invalid index")
            key_indices.append(int(raw_index))

        samples, maximum_gap = _arc_length_sample_positions(
            positions,
            uniform_sample_count=int(uniform_sample_count),
            key_position_indices=key_indices,
        )
        clearances = self.point_clearance(samples.astype(np.float32))
        sampled_minimum = float(np.min(clearances))
        return SampledPathCollisionReport(
            valid=bool(sampled_minimum >= required),
            minimum_clearance_m=sampled_minimum,
            required_clearance_m=required,
            sampled_position_count=len(samples),
            intersected_segment_indices=(),
            validation_contract="coarse_sampled_mesh_clearance",
            maximum_sample_spacing_m=maximum_gap,
            desired_continuous_clearance_m=required,
            required_sampled_clearance_m=required,
        )
