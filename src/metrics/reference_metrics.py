"""Reference-region appearance and native measured-depth comparisons."""
from __future__ import annotations

import numpy as np

from src.metrics.metrics import q0_return_metric, qe_evidence_metric, reproject_rgbd_to_target
from src.metrics.point_depth import evaluate_points
from src.metrics.transforms import canonical_to_output_transform, warp_image


class ReferenceMetrics:
    def __init__(self, case):
        self.case = case
        self.native_samples = case.source == 'ScanNet++'

    def transform(self, shape):
        height, width = shape
        return canonical_to_output_transform(
            canonical_width=self.case.canonical_camera.width,
            canonical_height=self.case.canonical_camera.height,
            output_width=width, output_height=height) @ self.case.native_to_canonical

    def install_depth(self, result, role, prediction, transform, branch='canonical', support=None):
        if prediction is None or branch not in result.raw:
            return result
        if result.raw[branch].get('status') == 'not_available':
            return result
        frame = self.case.q0 if role == 'q0' else self.case.qe
        masks = (self.case.q0_anchor_masks_rgb if role == 'q0'
                 else self.case.appearance_anchor_masks_rgb)
        region = None if role == 'q0' else self.case.appearance_support_rgb
        measured = evaluate_points(
            frame.depth_m, frame.depth_valid, frame.depth_camera.K,
            frame.rgb_camera.K, transform, prediction, masks,
            branch_support=support, native_rgb_support=region)
        raw = result.raw[branch]
        raw['depth'] = dict(measured['scene'], valid_pixel_count=measured['scene']['evaluated_point_count'])
        raw['anchor_depth'] = {
            'status': 'ok', 'anchors': measured['anchors'],
            'macro_abs_rel': measured['macro_abs_rel'],
            'parent_fitted_scale': measured['scene']['fitted_scale'],
            'scale_policy': 'shared_parent_no_anchor_refit', 'sampling': measured['sampling']}
        raw['reference_sampling'] = measured['sampling']
        return result

    def q0(self, **kwargs):
        result = q0_return_metric(**kwargs)
        if not self.native_samples:
            return result
        transform = self.transform(kwargs['q0_reference'].shape[:2])
        self.install_depth(result, 'q0', kwargs.get('canonical_depth'), transform)
        return self.install_depth(result, 'q0', kwargs.get('nearest_pose_depth'), transform, 'nearest_pose')

    def qe(self, **kwargs):
        transform = self.transform(kwargs['reference'].shape[:2])
        region = self.case.appearance_support_rgb
        if region is not None:
            height, width = kwargs['reference'].shape[:2]
            kwargs['input_supported_mask'] = warp_image(region.astype('uint8'), transform,
                                                       width=width, height=height, mask=True)
            kwargs['invalid_mask'] = np.zeros((height, width), dtype=bool)
            kwargs['anchor_masks'] = {
                name: warp_image(mask.astype('uint8'), transform,
                                 width=width, height=height, mask=True)
                for name, mask in self.case.appearance_anchor_masks_rgb.items()}
        result = qe_evidence_metric(**kwargs)
        if not self.native_samples:
            return result
        self.install_depth(result, 'qe', kwargs.get('canonical_depth'), transform)
        prediction = kwargs.get('nearest_pose_depth')
        support = None
        if (prediction is not None and kwargs.get('nearest_pose_frame') is not None
                and kwargs.get('nearest_pose_source_pose') is not None
                and kwargs.get('qe_target_pose') is not None and kwargs.get('camera_matrix') is not None):
            _, prediction, support = reproject_rgbd_to_target(
                kwargs['nearest_pose_frame'], prediction,
                source_pose_c2w=kwargs['nearest_pose_source_pose'],
                target_pose_c2w=kwargs['qe_target_pose'], camera_matrix=kwargs['camera_matrix'],
                target_shape=kwargs['reference'].shape[:2],
                source_depth_scale=kwargs.get('nearest_pose_geometry_scale', 1.0))
        self.install_depth(result, 'qe', prediction, transform, 'nearest_pose', support)
        result.raw['reference_appearance'] = 'q0_visible_qe'
        return result
