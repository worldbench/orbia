"""Metric families in the order used in the ORBIA paper."""
TIER1_METRICS = ("control",)
TIER2_METRICS = ("imaging_quality", "aesthetic_quality", "motion_smoothness")
TIER3_METRICS = ("q0_return", "evidence_query", "anchor_identity")
TIER4_METRICS = ("completion_revisit",)
TIER5_METRICS = ("depth_temporal_consistency", "covisible_reprojection", "regional_scale_inconsistency", "da3_reconstruction")
DEFAULT_METRICS = ("validity", *TIER1_METRICS, *TIER2_METRICS, *TIER3_METRICS, *TIER4_METRICS, *TIER5_METRICS)
KNOWN_METRICS = frozenset((*DEFAULT_METRICS, "long_horizon", "anchor_trajectory"))
# HPS, action labels, flicker and T6 profiles are computed by src.metrics.supporting.
