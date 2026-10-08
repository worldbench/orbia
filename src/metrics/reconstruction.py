"""Read recovered depth artifacts and export point-cloud diagnostics."""

from __future__ import annotations

from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np



def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency environment
        raise RuntimeError("OpenCV is required to decode ViPE depth") from exc
    return cv2


def _frame_index(path: Path, fallback: int) -> int:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    return int(match.group(1)) if match else fallback


def list_vipe_depth_indices(depth_root: Path | str) -> tuple[int, ...]:
    """Index ViPE depth artifacts without decoding depth pixels."""

    root = Path(depth_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    indices: list[int] = []
    files = sorted(path for path in root.iterdir() if path.is_file())
    for fallback, path in enumerate(files):
        suffix = path.suffix.casefold()
        if suffix in {".npy", ".npz", ".exr"}:
            indices.append(_frame_index(path, fallback))
        elif suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = [
                    name
                    for name in archive.namelist()
                    if Path(name).suffix.casefold() in {".npy", ".npz", ".exr"}
                ]
                indices.extend(
                    _frame_index(Path(name), member_fallback)
                    for member_fallback, name in enumerate(members)
                )
    if not indices:
        raise FileNotFoundError(f"no .npy/.npz/.exr depth found below {root}")
    if len(indices) != len(set(indices)):
        raise ValueError("ViPE depth frame indices must be unique")
    return tuple(sorted(indices))


def _decode_depth_bytes(name: str, payload: bytes) -> np.ndarray:
    suffix = Path(name).suffix.casefold()
    if suffix == ".npy":
        import io
        return np.load(io.BytesIO(payload), allow_pickle=False)
    if suffix == ".npz":
        import io
        archive = np.load(io.BytesIO(payload), allow_pickle=False)
        key = next((candidate for candidate in ("depth", "data", "arr_0") if candidate in archive), None)
        if key is None:
            raise ValueError(f"depth archive {name} has no recognised depth key")
        return archive[key]
    if suffix == ".exr":
        cv2 = _cv2()
        value = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if _has_positive_finite_depth(value):
            return value
        return _decode_openexr_bytes(name, payload)
    raise ValueError(f"unsupported depth member {name}")


def _load_depth_file(path: Path) -> np.ndarray:
    suffix = path.suffix.casefold()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        key = next((candidate for candidate in ("depth", "data", "arr_0") if candidate in archive), None)
        if key is None:
            raise ValueError(f"depth archive {path} has no recognised depth key")
        return archive[key]
    if suffix == ".exr":
        cv2 = _cv2()
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if _has_positive_finite_depth(value):
            return value
        return _decode_openexr_file(path)
    raise ValueError(f"unsupported depth file {path}")


def _has_positive_finite_depth(value: np.ndarray | None) -> bool:
    """Detect OpenCV builds that silently decode half-float EXR as zeros."""

    if value is None:
        return False
    array = np.asarray(value)
    return bool(np.any(np.isfinite(array) & (array > 0.0)))


def _decode_openexr_bytes(name: str, payload: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix) as stream:
        stream.write(payload)
        stream.flush()
        return _decode_openexr_file(Path(stream.name))


def _decode_openexr_file(path: Path) -> np.ndarray:
    """Decode the official ViPE ``Z`` channel when OpenCV is unavailable/broken."""

    try:
        import Imath
        import OpenEXR
    except ImportError as exc:  # pragma: no cover - dependency environment
        raise RuntimeError(
            "ViPE EXR depth needs OpenEXR when this OpenCV build cannot decode it; "
            "run with the ViPE environment or install OpenEXR."
        ) from exc
    image = OpenEXR.InputFile(str(path))
    header = image.header()
    channels = header["channels"]
    channel = "Z" if "Z" in channels else next(iter(channels), None)
    if channel is None:
        raise ValueError(f"EXR depth {path} has no channels")
    window = header["dataWindow"]
    width = window.max.x - window.min.x + 1
    height = window.max.y - window.min.y + 1
    raw = image.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
    value = np.frombuffer(raw, dtype=np.float32)
    if value.size != height * width:
        raise ValueError(f"EXR depth {path} has unexpected {channel} sample count")
    return value.reshape(height, width)


def load_vipe_depths(depth_root: Path | str) -> dict[int, np.ndarray]:
    """Read common official-ViPE depth layouts without modifying them."""

    root = Path(depth_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    values: dict[int, np.ndarray] = {}
    files = sorted(path for path in root.iterdir() if path.is_file())
    for fallback, path in enumerate(files):
        suffix = path.suffix.casefold()
        if suffix in {".npy", ".npz", ".exr"}:
            values[_frame_index(path, fallback)] = _load_depth_file(path)
        elif suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = [name for name in archive.namelist() if Path(name).suffix.casefold() in {".npy", ".npz", ".exr"}]
                for member_fallback, name in enumerate(members):
                    member_path = Path(name)
                    values[_frame_index(member_path, member_fallback)] = _decode_depth_bytes(name, archive.read(name))
    if not values:
        raise FileNotFoundError(f"no .npy/.npz/.exr depth found below {root}")
    return values


