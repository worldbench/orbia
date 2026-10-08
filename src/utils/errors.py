"""Explicit domain failures which may be rejected during sample selection."""

from __future__ import annotations


class OrbiaFeasibilityError(ValueError):
    """Base class for a scientifically valid but unusable candidate."""


class TrajectoryFeasibilityError(OrbiaFeasibilityError):
    """A candidate cannot realize the trajectory template."""


