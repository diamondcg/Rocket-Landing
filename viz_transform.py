"""Coordinate transform helpers for the simulation visualization.

Pure math only -- no SDL/OpenGL dependencies, so this module is safe to
import and unit test without a display or GL context.
"""

from __future__ import annotations

import numpy as np


def altitude_to_ndc_y(z: float, z_min: float, z_max: float) -> float:
    """Map an altitude to normalized device coordinates.

    Parameters
    ----------
    z:
        Altitude [m].
    z_min, z_max:
        Altitude range mapped to NDC ``[-1, 1]`` [m].

    Returns
    -------
    float
        Y coordinate in normalized device coordinates ``[-1, 1]``.
    """
    span = z_max - z_min
    frac = (z - z_min) / span
    return frac * 2.0 - 1.0


def altitude_to_screen_y(z: float, z_min: float, z_max: float,
                          screen_height: int, margin_px: int = 0) -> float:
    """Map an altitude to a screen-space y coordinate (origin top-left).

    Parameters
    ----------
    z:
        Altitude [m].
    z_min, z_max:
        Altitude range mapped to the drawable area [m].
    screen_height:
        Window height [px].
    margin_px:
        Top/bottom margin reserved on screen [px].

    Returns
    -------
    float
        Pixel y coordinate (``0`` = top of window, increases downward).
        Higher altitudes map to smaller y values.
    """
    drawable = screen_height - 2 * margin_px
    span = z_max - z_min
    frac = (z - z_min) / span
    return margin_px + (1.0 - frac) * drawable


def world_to_ndc(x: float, y: float, x_range: tuple[float, float],
                  y_range: tuple[float, float]) -> tuple[float, float]:
    """Map a 2D world-space point to normalized device coordinates.

    Parameters
    ----------
    x, y:
        World-space coordinates.
    x_range, y_range:
        ``(min, max)`` world-space ranges mapped to NDC ``[-1, 1]``.

    Returns
    -------
    tuple[float, float]
        ``(x_ndc, y_ndc)`` in ``[-1, 1]``.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_ndc = (x - x_min) / (x_max - x_min) * 2.0 - 1.0
    y_ndc = (y - y_min) / (y_max - y_min) * 2.0 - 1.0
    return x_ndc, y_ndc


def compute_view_bounds(z_array: np.ndarray,
                         padding_frac: float = 0.1) -> tuple[float, float]:
    """Compute altitude view bounds from a recorded altitude array.

    Parameters
    ----------
    z_array:
        Recorded altitude samples [m].
    padding_frac:
        Fractional padding applied above the maximum altitude, relative to
        the altitude span.

    Returns
    -------
    tuple[float, float]
        ``(z_min, z_max)`` view bounds [m]. ``z_min`` is always ``<= 0`` so
        the ground is included in the view.
    """
    z_min = min(0.0, float(np.min(z_array)))
    z_max = float(np.max(z_array))

    span = z_max - z_min
    if span == 0.0:
        span = 1.0

    z_max += span * padding_frac

    return z_min, z_max
