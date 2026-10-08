"""Independent ScanNet geometry sampler; no sparse-mask raster resizing."""
import numpy as np


def map_points(depth_shape, depth_K, rgb_K, native_to_output):
    y, x = np.indices(depth_shape)
    rays = np.stack([x.ravel()+.5, y.ravel()+.5, np.ones(x.size)])
    uv = np.asarray(rgb_K) @ np.linalg.inv(depth_K) @ rays
    # Convert camera pixel centres into the array coordinates used by warpPerspective.
    native_xy = (uv[:2] / uv[2]).T - .5
    mapped = np.asarray(native_to_output) @ np.column_stack(
        [native_xy, np.ones(len(native_xy))]).T
    with np.errstate(divide='ignore', invalid='ignore'):
        output_xy = (mapped[:2] / mapped[2]).T
    return native_xy, output_xy


def sample_nearest(array, xy):
    """One lookup per original point; no dilation, interpolation or duplication."""
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError('Expected a two-dimensional map')
    xy = np.asarray(xy)
    h, w = array.shape
    inside = np.isfinite(xy).all(axis=1)
    inside &= (xy[:, 0] >= -.5) & (xy[:, 0] < w-.5)
    inside &= (xy[:, 1] >= -.5) & (xy[:, 1] < h-.5)
    ij = np.zeros(xy.shape, dtype=np.int64)
    ij[inside] = np.floor(xy[inside]+.5).astype(np.int64)
    values = np.zeros(len(xy), dtype=array.dtype)
    values[inside] = array[ij[inside, 1], ij[inside, 0]]
    return values, inside, ij


def evaluate_points(reference_depth, expected_depth_mask, depth_K, rgb_K,
                    native_to_output, prediction, anchor_masks_native,
                    branch_support=None, native_rgb_support=None):
    """Scene AbsRel plus anchor AbsRel with ONE inherited scene median scale."""
    reference = np.asarray(reference_depth, dtype=float)
    expected = np.asarray(expected_depth_mask, dtype=bool)
    prediction = np.asarray(prediction, dtype=float)
    if reference.ndim != 2 or expected.shape != reference.shape:
        raise ValueError('Reference and support must share the native depth grid')
    if branch_support is not None and np.shape(branch_support) != prediction.shape:
        raise ValueError('Branch support must be aligned to prediction')
    native_xy, output_xy = map_points(reference.shape, depth_K, rgb_K, native_to_output)
    pred, inside, ij = sample_nearest(prediction, output_xy)
    ref = reference.ravel()
    source = expected.ravel() & np.isfinite(ref) & (ref > 0)
    if native_rgb_support is not None:
        rgb_supported, rgb_inside, _ = sample_nearest(
            np.asarray(native_rgb_support, bool), native_xy)
        source &= rgb_inside & rgb_supported
    eligible = source & inside
    common = eligible & np.isfinite(pred) & (pred > 0)
    if branch_support is not None:
        support, _, _ = sample_nearest(np.asarray(branch_support, bool), output_xy)
        common &= support
    scale = float(np.median(ref[common] / pred[common])) if common.any() else None

    def result(mask):
        available = eligible & mask
        used = common & mask
        return {
            'source_point_count': int((source & mask).sum()),
            'in_output_point_count': int(available.sum()),
            'evaluated_point_count': int(used.sum()),
            'unique_prediction_pixels': int(len(np.unique(ij[used], axis=0))),
            'coverage': float(used.sum()/available.sum()) if available.any() else None,
            'abs_rel': float(np.mean(np.abs(scale*pred[used]-ref[used])/ref[used]))
                       if used.any() and scale is not None else None,
        }
    scene = result(np.ones(len(ref), bool))
    scene['fitted_scale'] = scale
    anchors = {}
    for name, mask in anchor_masks_native.items():
        membership, valid, _ = sample_nearest(np.asarray(mask, bool), native_xy)
        anchors[name] = result(membership & valid)
    scores = [a['abs_rel'] for a in anchors.values() if a['abs_rel'] is not None]
    return {'scene': scene, 'anchors': anchors,
            'macro_abs_rel': float(np.mean(scores)) if scores else None,
            'sampling': 'native_reference_points_nearest_prediction',
            'scale_scope': 'one_scene_scale_shared_by_all_anchors'}
