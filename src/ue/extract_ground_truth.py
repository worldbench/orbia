#!/usr/bin/env python3
"""Extract aligned RGB, metric Z-depth, and integer masks from UE MRQ EXRs."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


FRAME_COUNT = 357
FPS = 16
WIDTH = 1280
HEIGHT = 720
EXPECTED_STEMS = [f"{index:06d}" for index in range(FRAME_COUNT)]


@dataclass(frozen=True)
class ExrSpec:
    width: int
    height: int
    channels: tuple[str, ...]
    metadata: dict[str, Any]


def _load_oiio():
    try:
        import OpenImageIO as oiio  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "OpenImageIO is required for multilayer EXR/Cryptomatte input. "
            "Install ORBIA UE extras: python3 -m pip install 'orbia[ue]'"
        ) from exc
    return oiio


def _metadata_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def _open_spec(path: Path, oiio: Any) -> ExrSpec:
    image = oiio.ImageInput.open(str(path))
    if image is None:
        raise ValueError(f"OpenImageIO could not open {path}: {oiio.geterror()}")
    try:
        spec = image.spec()
        metadata = {
            str(attribute.name): _metadata_value(attribute.value)
            for attribute in spec.extra_attribs
        }
        return ExrSpec(
            width=int(spec.width),
            height=int(spec.height),
            channels=tuple(str(name) for name in spec.channelnames),
            metadata=metadata,
        )
    finally:
        image.close()


def _read_pixels(path: Path, oiio: Any) -> tuple[ExrSpec, np.ndarray]:
    image = oiio.ImageInput.open(str(path))
    if image is None:
        raise ValueError(f"OpenImageIO could not open {path}: {oiio.geterror()}")
    try:
        spec = image.spec()
        metadata = {
            str(attribute.name): _metadata_value(attribute.value)
            for attribute in spec.extra_attribs
        }
        try:
            pixels = image.read_image(format=oiio.FLOAT)
        except TypeError:
            pixels = image.read_image()
        if pixels is None:
            raise ValueError(f"Could not read pixels from {path}: {image.geterror()}")
        array = np.asarray(pixels, dtype=np.float32)
        expected = int(spec.width) * int(spec.height) * int(spec.nchannels)
        if array.size != expected:
            raise ValueError(
                f"Unexpected pixel count in {path}: {array.size}, expected {expected}"
            )
        array = array.reshape(int(spec.height), int(spec.width), int(spec.nchannels))
        return (
            ExrSpec(
                width=int(spec.width),
                height=int(spec.height),
                channels=tuple(str(name) for name in spec.channelnames),
                metadata=metadata,
            ),
            array,
        )
    finally:
        image.close()


def _suffix(name: str) -> tuple[str, str]:
    if "." not in name:
        return "", name.lower()
    prefix, component = name.rsplit(".", 1)
    aliases = {"red": "r", "green": "g", "blue": "b", "alpha": "a"}
    component = aliases.get(component.lower(), component.lower())
    return prefix, component


def _rgb_indices(channels: tuple[str, ...]) -> tuple[int, int, int]:
    groups: dict[str, dict[str, int]] = {}
    for index, name in enumerate(channels):
        prefix, component = _suffix(name)
        if component in {"r", "g", "b"}:
            groups.setdefault(prefix, {})[component] = index
    candidates = []
    for prefix, components in groups.items():
        if not {"r", "g", "b"}.issubset(components):
            continue
        lowered = prefix.lower()
        if any(word in lowered for word in ("depth", "motion", "velocity", "mask")):
            continue
        score = 2 if "finalimage" in lowered else 1 if not prefix else 0
        candidates.append((score, prefix, components))
    if not candidates:
        raise ValueError(f"Could not identify FinalImage RGB channels in {channels}")
    _, _, chosen = max(candidates, key=lambda item: (item[0], item[1]))
    return chosen["r"], chosen["g"], chosen["b"]


def _depth_index(channels: tuple[str, ...]) -> int:
    candidates = []
    for index, name in enumerate(channels):
        prefix, component = _suffix(name)
        lowered = name.lower()
        if "worlddepth" in lowered and component in {"r", "z"}:
            candidates.append((0 if "movierenderqueue_worlddepth" in lowered else 1, index))
        elif "worlddepth" in lowered and not prefix:
            candidates.append((2, index))
    if not candidates:
        raise ValueError(f"Could not identify MovieRenderQueue_WorldDepth in {channels}")
    return min(candidates)[1]


def _cryptomatte_manifests(metadata: dict[str, Any]) -> list[tuple[str, dict[str, str]]]:
    manifests = []
    for key, value in metadata.items():
        if not key.lower().startswith("cryptomatte/") or not key.lower().endswith("/manifest"):
            continue
        token = key.split("/")[1]
        try:
            payload = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Cryptomatte manifest in EXR metadata {key}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Cryptomatte manifest {key} is not an object")
        name = str(metadata.get(f"cryptomatte/{token}/name", ""))
        manifests.append((name, {str(k): str(v) for k, v in payload.items()}))
    return manifests


def _crypto_groups(
    channels: tuple[str, ...], metadata: dict[str, Any]
) -> tuple[dict[str, str], list[tuple[int, int]]]:
    manifests = _cryptomatte_manifests(metadata)
    if not manifests:
        raise ValueError("EXR has no Cryptomatte manifest; Object IDs pass is missing")

    def matching_channels(pass_name: str) -> list[tuple[int, int]]:
        groups: dict[str, dict[str, int]] = {}
        for index, channel in enumerate(channels):
            prefix, component = _suffix(channel)
            lowered = prefix.lower()
            looks_like_object_id = any(
                word in lowered for word in ("hitproxymask", "objectid", "cryptomatte")
            )
            if pass_name:
                looks_like_object_id = pass_name.lower() in lowered
            if looks_like_object_id and component in {"r", "g", "b", "a"}:
                groups.setdefault(prefix, {})[component] = index
        pairs = []
        for prefix in sorted(groups):
            components = groups[prefix]
            if "r" in components and "g" in components:
                pairs.append((components["r"], components["g"]))
            if "b" in components and "a" in components:
                pairs.append((components["b"], components["a"]))
        return pairs

    options = [(manifest, matching_channels(name)) for name, manifest in manifests]
    options = [option for option in options if option[1]]
    if not options:
        raise ValueError(
            "Found Cryptomatte metadata but could not match its RGBA layers to EXR channels"
        )
    if len(options) > 1:
        raise ValueError("EXR contains multiple usable Cryptomatte passes; expected one Object ID pass")
    return options[0]


def _hash_int(text: str) -> int:
    value = text.strip().lower()
    if value.startswith("0x"):
        value = value[2:]
    if len(value) > 8:
        raise ValueError(f"Cryptomatte hash is wider than uint32: {text}")
    return int(value, 16)


def _manifest_union(specs: list[ExrSpec]) -> dict[str, int]:
    names_to_hashes: dict[str, int] = {}
    hashes_to_names: dict[int, str] = {}
    for spec in specs:
        manifests = _cryptomatte_manifests(spec.metadata)
        if len(manifests) != 1:
            raise ValueError(
                f"Expected exactly one Cryptomatte manifest, found {len(manifests)}"
            )
        for name, hash_text in manifests[0][1].items():
            hash_value = _hash_int(hash_text)
            old_hash = names_to_hashes.setdefault(name, hash_value)
            if old_hash != hash_value:
                raise ValueError(f"Cryptomatte name {name!r} changed hash across frames")
            old_name = hashes_to_names.setdefault(hash_value, name)
            if old_name != name:
                raise ValueError(
                    f"Cryptomatte uint32 collision between {old_name!r} and {name!r}"
                )
    return names_to_hashes


def _decode_mask(
    pixels: np.ndarray,
    pairs: list[tuple[int, int]],
    hash_to_instance: dict[int, int],
) -> tuple[np.ndarray, set[int]]:
    height, width, _ = pixels.shape
    winning_hash = np.zeros((height, width), dtype=np.uint32)
    winning_coverage = np.zeros((height, width), dtype=np.float32)
    for id_index, coverage_index in pairs:
        ids = np.ascontiguousarray(pixels[:, :, id_index], dtype=np.float32).view(np.uint32)
        coverage = pixels[:, :, coverage_index]
        replace = np.isfinite(coverage) & (coverage > winning_coverage) & (ids != 0)
        winning_hash[replace] = ids[replace]
        winning_coverage[replace] = coverage[replace]

    keys = np.asarray(sorted(hash_to_instance), dtype=np.uint32)
    values = np.asarray([hash_to_instance[int(key)] for key in keys], dtype=np.uint32)
    flat = winning_hash.ravel()
    mask_flat = np.zeros(flat.shape, dtype=np.uint32)
    unknown = set()
    foreground = flat != 0
    if foreground.any():
        if not len(keys):
            unknown = {int(value) for value in np.unique(flat[foreground])}
        else:
            positions = np.searchsorted(keys, flat[foreground])
            clipped = np.minimum(positions, len(keys) - 1)
            valid = (positions < len(keys)) & (keys[clipped] == flat[foreground])
            target = np.flatnonzero(foreground)
            mask_flat[target[valid]] = values[clipped[valid]]
            unknown = {
                int(value)
                for value in np.unique(flat[target[~valid]])
            }
    return mask_flat.reshape(height, width), unknown


def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def _intrinsics_from_frame(frame: dict) -> tuple[float, float, float, float]:
    width, height = map(int, frame["resolution_px"])
    if (width, height) != (WIDTH, HEIGHT):
        raise ValueError(f"Camera resolution is {width}x{height}, expected {WIDTH}x{HEIGHT}")
    focal = float(frame["focal_length_mm"])
    fx = focal / float(frame["sensor_width_mm"]) * width
    fy = focal / float(frame["sensor_height_mm"]) * height
    return fx, fy, width / 2.0, height / 2.0


def _validate_trajectory(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("frame_count") != FRAME_COUNT or payload.get("fps") != FPS:
        raise ValueError(f"{path} must use the fixed {FRAME_COUNT}-frame/{FPS}-FPS contract")
    frames = payload.get("frames", [])
    indices = [frame.get("frame_index") for frame in frames]
    if indices != list(range(FRAME_COUNT)):
        raise ValueError(f"{path} must contain consecutive frame indices 0..{FRAME_COUNT - 1}")


def _to_z_depth(
    raw: np.ndarray,
    definition: str,
    unit_scale: float,
    intrinsics: tuple[float, float, float, float] | None,
    max_depth_m: float,
) -> np.ndarray:
    depth_m = np.asarray(raw, dtype=np.float32) * np.float32(unit_scale)
    # Reject source-pass sentinels before ray-to-Z conversion.  In UE 5.7 the
    # WorldDepth post-process pass uses the half-float maximum (65504 cm) for
    # uncovered/sky pixels; testing only after division by ray norm can turn
    # that sentinel into an apparently valid edge-pixel depth.
    invalid = (~np.isfinite(depth_m) | (depth_m <= 0.0)
               | (depth_m > max_depth_m) | (np.asarray(raw) == 65504.0))
    if definition == "ray_distance":
        if intrinsics is None:
            raise ValueError("ray_distance conversion requires camera_frames.json")
        fx, fy, cx, cy = intrinsics
        x = (np.arange(depth_m.shape[1], dtype=np.float32) + 0.5 - cx) / fx
        y = (np.arange(depth_m.shape[0], dtype=np.float32) + 0.5 - cy) / fy
        ray_norm = np.sqrt(1.0 + y[:, None] ** 2 + x[None, :] ** 2)
        depth_m = depth_m / ray_norm
    invalid |= ~np.isfinite(depth_m) | (depth_m <= 0.0) | (depth_m > max_depth_m)
    depth_m[invalid] = np.nan
    return depth_m


def _frame_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    by_stem = {
        path.stem: path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == ".exr"
    }
    missing = sorted(set(EXPECTED_STEMS) - set(by_stem))
    extra = sorted(set(by_stem) - set(EXPECTED_STEMS))
    if missing or extra:
        raise ValueError(
            f"Input must contain {FRAME_COUNT} consecutive EXRs; "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    return [by_stem[stem] for stem in EXPECTED_STEMS]


def _prepare_outputs(capture: Path) -> tuple[Path, Path, Path]:
    outputs = capture / "rgb", capture / "depth", capture / "instance_mask"
    conflicts = [path for path in outputs if path.exists() and any(path.iterdir())]
    metadata_conflicts = [
        path for path in (capture / "instances.json", capture / "capture_manifest.json")
        if path.exists()
    ]
    if conflicts or metadata_conflicts:
        names = ", ".join(str(path) for path in conflicts + metadata_conflicts)
        raise ValueError(f"Refusing to overwrite existing derived capture data: {names}")
    for path in outputs:
        path.mkdir(parents=True, exist_ok=True)
    return outputs


def _copy_new(source: Path | None, destination: Path) -> None:
    if source is None:
        return
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and source != destination.resolve():
        if source.read_bytes() == destination.read_bytes():
            return
        raise ValueError(f"Refusing to overwrite {destination}")
    if source != destination.resolve():
        shutil.copy2(source, destination)


def _write_mask(path: Path, mask: np.ndarray, instance_count: int) -> str:
    if instance_count <= 255:
        Image.fromarray(mask.astype(np.uint8)).save(path)
        return "uint8 PNG; 0=background"
    if instance_count <= 65535:
        Image.fromarray(mask.astype(np.uint16)).save(path)
        return "uint16 PNG; 0=background"
    raise ValueError("More than 65535 Object IDs cannot be represented by the PNG contract")


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract ORBIA ground truth from UE multilayer MRQ EXRs"
    )
    parser.add_argument("--input", type=Path, required=True, help="directory of consecutive multilayer EXRs")
    parser.add_argument("--capture", type=Path, required=True, help="raw_capture output root")
    parser.add_argument("--camera-frames", type=Path, help="camera_frames.json to copy/use")
    parser.add_argument("--trajectory", type=Path, help="trajectory_spec.json to copy")
    parser.add_argument(
        "--world-depth-definition",
        choices=("camera_axis_z", "ray_distance"),
        required=True,
        help="meaning verified for MovieRenderQueue_WorldDepth in your UE calibration map",
    )
    parser.add_argument("--world-unit-to-m", type=float, default=0.01)
    parser.add_argument("--max-depth-m", type=float, default=650.0)
    parser.add_argument(
        "--rgb-transfer",
        choices=("linear_to_srgb", "identity"),
        default="linear_to_srgb",
        help="EXR FinalImage transfer before 8-bit PNG quantization",
    )
    parser.add_argument(
        "--allow-unknown-ids",
        action="store_true",
        help="map Cryptomatte hashes absent from the manifest to background",
    )
    parser.add_argument(
        "--manifest-overrides",
        type=Path,
        help="JSON fields merged into capture_manifest.json (asset provenance and seed)",
    )
    args = parser.parse_args(argv)

    global FRAME_COUNT, FPS, WIDTH, HEIGHT, EXPECTED_STEMS
    trajectory_path = args.trajectory or args.capture / "trajectory_spec.json"
    camera_path = args.camera_frames or args.capture / "camera_frames.json"
    trajectory_config = json.loads(trajectory_path.read_text(encoding="utf-8"))
    camera_config = json.loads(camera_path.read_text(encoding="utf-8"))
    measured_frames = camera_config.get("frames", camera_config) if isinstance(camera_config, dict) else camera_config
    FRAME_COUNT = int(trajectory_config["frame_count"])
    FPS = int(trajectory_config["fps"])
    WIDTH, HEIGHT = map(int, measured_frames[0]["resolution_px"])
    EXPECTED_STEMS = [f"{i:06d}" for i in range(FRAME_COUNT)]

    if args.world_unit_to_m <= 0.0 or args.max_depth_m <= 0.0:
        parser.error("depth scale and max depth must be positive")

    oiio = _load_oiio()
    input_root = args.input.expanduser().resolve()
    capture = args.capture.expanduser().resolve()
    files = _frame_files(input_root)
    specs = [_open_spec(path, oiio) for path in files]
    for path, spec in zip(files, specs):
        if (spec.width, spec.height) != (WIDTH, HEIGHT):
            raise ValueError(
                f"{path} is {spec.width}x{spec.height}; expected {WIDTH}x{HEIGHT}"
            )
        _rgb_indices(spec.channels)
        _depth_index(spec.channels)
        _crypto_groups(spec.channels, spec.metadata)

    names_to_hashes = _manifest_union(specs)
    ordered_names = sorted(names_to_hashes)
    if len(ordered_names) > 65535:
        raise ValueError(f"Found {len(ordered_names)} Object IDs; maximum supported is 65535")
    name_to_instance = {name: index for index, name in enumerate(ordered_names, start=1)}
    hash_to_instance = {
        names_to_hashes[name]: name_to_instance[name] for name in ordered_names
    }

    camera_source = args.camera_frames
    if camera_source is None and (capture / "camera_frames.json").is_file():
        camera_source = capture / "camera_frames.json"
    if camera_source is None:
        raise ValueError(
            "--camera-frames is required unless camera_frames.json already exists in --capture"
        )
    camera_source = camera_source.expanduser().resolve()
    if len(measured_frames) != FRAME_COUNT:
        raise ValueError("Camera metadata and trajectory frame counts disagree")
    camera_intrinsics_all = [_intrinsics_from_frame(frame) for frame in measured_frames]
    camera_intrinsics = (
        camera_intrinsics_all if args.world_depth_definition == "ray_distance" else None
    )

    trajectory_source = args.trajectory
    if trajectory_source is None and (capture / "trajectory_spec.json").is_file():
        trajectory_source = capture / "trajectory_spec.json"
    if trajectory_source is None:
        raise ValueError(
            "--trajectory is required unless trajectory_spec.json already exists in --capture"
        )
    trajectory_source = trajectory_source.expanduser().resolve()
    _validate_trajectory(trajectory_source)

    rgb_dir, depth_dir, mask_dir = _prepare_outputs(capture)
    mask_encoding = ""
    for frame_index, path in enumerate(files):
        spec, pixels = _read_pixels(path, oiio)
        r, g, b = _rgb_indices(spec.channels)
        rgb = pixels[:, :, [r, g, b]]
        if args.rgb_transfer == "linear_to_srgb":
            rgb = _srgb_encode(rgb)
        else:
            rgb = np.clip(rgb, 0.0, 1.0)
        rgb8 = np.rint(rgb * 255.0).astype(np.uint8)
        Image.fromarray(rgb8).save(rgb_dir / f"{frame_index:06d}.png")

        depth = _to_z_depth(
            pixels[:, :, _depth_index(spec.channels)],
            args.world_depth_definition,
            args.world_unit_to_m,
            camera_intrinsics[frame_index] if camera_intrinsics is not None else None,
            args.max_depth_m,
        )
        np.save(depth_dir / f"{frame_index:06d}.npy", depth)

        _, pairs = _crypto_groups(spec.channels, spec.metadata)
        mask, unknown = _decode_mask(pixels, pairs, hash_to_instance)
        # Cryptomatte may assign the engine's synthetic default actor to sky or
        # otherwise uncovered pixels.  ID 0 is reserved for
        # background, which is exactly the set of pixels with invalid depth.
        mask[~np.isfinite(depth)] = 0
        if unknown and not args.allow_unknown_ids:
            sample = ", ".join(f"0x{value:08x}" for value in sorted(unknown)[:8])
            raise ValueError(
                f"{path} contains Object-ID hashes absent from its Cryptomatte manifest: {sample}"
            )
        mask_encoding = _write_mask(
            mask_dir / f"{frame_index:06d}.png", mask, len(ordered_names)
        )
        if frame_index % 25 == 0 or frame_index == FRAME_COUNT - 1:
            print(f"extracted {frame_index + 1}/{FRAME_COUNT}: {path.name}")

    instances = {
        str(name_to_instance[name]): {
            "actor_label": name.rsplit("/", 1)[-1],
            "actor_path": name,
            "semantic_category": "unknown",
            "cryptomatte_hash_hex": f"{names_to_hashes[name]:08x}",
        }
        for name in ordered_names
    }
    (capture / "instances.json").write_text(
        json.dumps(
            {
                "contract": "orbia.unreal.instances.v1",
                "background_id": 0,
                "encoding": mask_encoding,
                "source": "MRQ Object IDs Cryptomatte; maximum-coverage hard assignment",
                "instances": instances,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    _copy_new(camera_source, capture / "camera_frames.json")
    _copy_new(trajectory_source, capture / "trajectory_spec.json")
    partial = _load_json_if_present(capture / "capture_manifest.partial.json")
    overrides = (
        _load_json_if_present(args.manifest_overrides)
        if args.manifest_overrides is not None
        else {}
    )
    manifest = {
        **partial,
        **overrides,
        "contract": "orbia.unreal.capture.v1",
        "resolution_px": [WIDTH, HEIGHT],
        "fps": FPS,
        "frame_count": FRAME_COUNT,
        "rgb_encoding": f"8-bit PNG; transfer={args.rgb_transfer}",
        "depth_encoding": "float32 NumPy; invalid pixels are NaN",
        "depth_definition": "OpenCV camera-axis Z",
        "depth_unit": "metres",
        "source_world_depth_definition": args.world_depth_definition,
        "source_world_unit_to_m": args.world_unit_to_m,
        "instance_mask_encoding": mask_encoding,
        "instance_count": len(ordered_names),
        "source_exr_directory": str(input_root),
        "extraction": {
            "cryptomatte_assignment": "highest coverage; ties keep first channel pair",
            "max_valid_depth_m": args.max_depth_m,
            "unknown_object_ids_allowed": bool(args.allow_unknown_ids),
        },
    }
    (capture / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(capture / "capture_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
