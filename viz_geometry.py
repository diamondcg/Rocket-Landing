"""Vertex-generation helpers for the simulation visualization.

Pure numpy only -- no SDL/OpenGL dependencies, so this module is safe to
import and unit test without a display or GL context. All vertices are
returned in normalized device coordinates (NDC), ``[-1, 1]`` on each axis.
"""

from __future__ import annotations

import numpy as np

from viz_transform import altitude_to_ndc_y


def rocket_body_vertices(center_x: float, center_y_ndc: float,
                          width: float, height: float) -> np.ndarray:
    """Generate NDC vertices for a simple rocket polygon.

    The rocket is drawn as a house-shaped pentagon: a rectangular body with
    a triangular nose on top, centered at ``(center_x, center_y_ndc)``.

    Parameters
    ----------
    center_x:
        X coordinate of the rocket center, in NDC.
    center_y_ndc:
        Y coordinate of the rocket center, in NDC.
    width:
        Total width of the rocket body, in NDC units.
    height:
        Total height of the rocket (body + nose), in NDC units.

    Returns
    -------
    np.ndarray
        Shape ``(5, 2)`` array of ``(x, y)`` NDC vertices, ordered for
        ``GL_TRIANGLE_FAN`` or ``GL_LINE_LOOP``.
    """
    half_w = width / 2.0
    half_h = height / 2.0
    nose_h = min(width, height)
    shoulder_y = center_y_ndc + half_h - nose_h

    return np.array([
        [center_x, center_y_ndc + half_h],          # nose tip
        [center_x + half_w, shoulder_y],            # right shoulder
        [center_x + half_w, center_y_ndc - half_h],  # bottom right
        [center_x - half_w, center_y_ndc - half_h],  # bottom left
        [center_x - half_w, shoulder_y],            # left shoulder
    ])


def segmented_rocket_body_vertices(
    center_x: float, center_y_ndc: float, width: float, height: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate NDC vertices for a 3-segment shuttle-style rocket body.

    The rocket is split into ``forward``, ``mid``, and ``aft`` fuselage
    segments stacked vertically. ``forward`` carries the triangular nose
    (same construction as :func:`rocket_body_vertices`); ``mid`` and
    ``aft`` are rectangular quads. Together the three segments tile the
    same bounding box as ``rocket_body_vertices(center_x, center_y_ndc,
    width, height)``.

    Parameters
    ----------
    center_x:
        X coordinate of the rocket center, in NDC.
    center_y_ndc:
        Y coordinate of the rocket center, in NDC.
    width:
        Total width of the rocket body, in NDC units.
    height:
        Total height of the rocket (body + nose), in NDC units.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(forward, mid, aft)``. ``forward`` is shape ``(5, 2)``
        (pentagon: nose tip + 4 corners). ``mid`` and ``aft`` are shape
        ``(4, 2)`` rectangles. All ordered for ``GL_TRIANGLE_FAN`` or
        ``GL_LINE_LOOP``.
    """
    half_w = width / 2.0
    half_h = height / 2.0
    nose_h = min(width, height)
    shoulder_y = center_y_ndc + half_h - nose_h
    body_bottom = center_y_ndc - half_h

    seg_h = max(shoulder_y - body_bottom, 1e-6) / 3.0
    mid_y = body_bottom + seg_h
    forward_y = body_bottom + 2 * seg_h

    aft = np.array([
        [center_x - half_w, body_bottom],
        [center_x + half_w, body_bottom],
        [center_x + half_w, mid_y],
        [center_x - half_w, mid_y],
    ])

    mid = np.array([
        [center_x - half_w, mid_y],
        [center_x + half_w, mid_y],
        [center_x + half_w, forward_y],
        [center_x - half_w, forward_y],
    ])

    forward = np.array([
        [center_x, center_y_ndc + half_h],          # nose tip
        [center_x + half_w, shoulder_y],            # right shoulder
        [center_x + half_w, forward_y],             # bottom right
        [center_x - half_w, forward_y],             # bottom left
        [center_x - half_w, shoulder_y],            # left shoulder
    ])

    return forward, mid, aft


def engine_nozzle_vertices(center_x: float, body_bottom_y: float,
                            body_width: float, engine_height: float,
                            n_engines: int = 2, gap_frac: float = 0.15
                            ) -> list[np.ndarray]:
    """Generate NDC vertices for tapered engine-bell polygons.

    The engines are evenly spaced trapezoids hanging below the aft
    segment's bottom edge, each wider at the top (flush with the fuselage)
    and narrower at the bottom (nozzle exit), giving a rocket-engine
    silhouette.

    Parameters
    ----------
    center_x:
        X coordinate of the rocket center, in NDC.
    body_bottom_y:
        Y coordinate of the aft segment's bottom edge, in NDC. The engines
        hang below this.
    body_width:
        Total width of the rocket body, in NDC units. The engines are
        spaced evenly within this width.
    engine_height:
        Height of each engine nozzle, in NDC units.
    n_engines:
        Number of engine nozzles to generate.
    gap_frac:
        Fraction of each engine's slot width left as a gap between/around
        nozzles.

    Returns
    -------
    list[np.ndarray]
        ``n_engines`` shape ``(4, 2)`` arrays, each a trapezoid (top-left,
        top-right, bottom-right, bottom-left), ordered for
        ``GL_TRIANGLE_FAN``/``GL_LINE_LOOP``.
    """
    slot_width = body_width / n_engines
    top_width = slot_width * (1.0 - gap_frac)
    bottom_width = top_width * 0.6
    bottom_y = body_bottom_y - engine_height

    left_edge = center_x - body_width / 2.0

    nozzles = []
    for i in range(n_engines):
        slot_center = left_edge + slot_width * (i + 0.5)
        top_half = top_width / 2.0
        bottom_half = bottom_width / 2.0
        nozzles.append(np.array([
            [slot_center - top_half, body_bottom_y],
            [slot_center + top_half, body_bottom_y],
            [slot_center + bottom_half, bottom_y],
            [slot_center - bottom_half, bottom_y],
        ]))

    return nozzles


def flame_vertices(center_x: float, base_y_ndc: float, thrust_frac: float,
                    width: float, max_height: float) -> np.ndarray:
    """Generate NDC vertices for an exhaust-flame triangle.

    Parameters
    ----------
    center_x:
        X coordinate of the flame center, in NDC.
    base_y_ndc:
        Y coordinate of the flame's base (the rocket's exhaust point), in
        NDC.
    thrust_frac:
        Current thrust as a fraction of maximum thrust, in ``[0, 1]``.
        The flame's length scales linearly with this value.
    width:
        Width of the flame's base, in NDC units.
    max_height:
        Length of the flame at ``thrust_frac == 1``, in NDC units.

    Returns
    -------
    np.ndarray
        Shape ``(3, 2)`` array of ``(x, y)`` NDC vertices, ordered for
        ``GL_TRIANGLE_FAN`` or ``GL_LINE_LOOP``.
    """
    half_w = width / 2.0
    tip_y = base_y_ndc - max_height * thrust_frac

    return np.array([
        [center_x - half_w, base_y_ndc],
        [center_x + half_w, base_y_ndc],
        [center_x, tip_y],
    ])


def ground_line_vertices(y_ndc: float, x_extent: float = 1.0) -> np.ndarray:
    """Generate NDC vertices for the ground line.

    Parameters
    ----------
    y_ndc:
        Y coordinate of the ground line, in NDC (typically ``z = 0``
        mapped through :func:`viz_transform.altitude_to_ndc_y`).
    x_extent:
        Half-width of the line, in NDC units.

    Returns
    -------
    np.ndarray
        Shape ``(2, 2)`` array of ``(x, y)`` NDC endpoints, ordered for
        ``GL_LINES``.
    """
    return np.array([
        [-x_extent, y_ndc],
        [x_extent, y_ndc],
    ])


def arrow_vertices(origin_x: float, origin_y: float, magnitude: float,
                    max_magnitude: float, max_length: float, direction: int,
                    shaft_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate NDC vertices for a vertical force arrow.

    The arrow's shaft starts at ``(origin_x, origin_y)`` and extends
    vertically by an amount proportional to ``magnitude``, capped at
    ``max_length``.

    Parameters
    ----------
    origin_x, origin_y:
        Arrow base position, in NDC.
    magnitude:
        Force magnitude (assumed non-negative) [N].
    max_magnitude:
        Magnitude at which the arrow reaches ``max_length`` [N].
    max_length:
        Arrow length at ``magnitude >= max_magnitude``, in NDC units.
    direction:
        ``+1`` to draw the arrow pointing up, ``-1`` to draw it pointing
        down.
    shaft_width:
        Used to size the arrowhead, in NDC units.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(shaft, head)``. ``shaft`` is a shape ``(2, 2)`` array of line
        endpoints for ``GL_LINES``. ``head`` is a shape ``(3, 2)`` array of
        triangle vertices for ``GL_TRIANGLE_FAN``/``GL_LINE_LOOP``.
    """
    frac = max(0.0, min(magnitude / max_magnitude, 1.0))
    length = frac * max_length
    tip_y = origin_y + direction * length

    shaft = np.array([
        [origin_x, origin_y],
        [origin_x, tip_y],
    ])

    head_size = shaft_width * 3.0
    base_y = tip_y - direction * head_size
    head = np.array([
        [origin_x, tip_y],
        [origin_x - head_size, base_y],
        [origin_x + head_size, base_y],
    ])

    return shaft, head


def progress_bar_vertices(x: float, y: float, width: float, height: float,
                           frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate NDC vertices for a vertical progress-bar gauge.

    Parameters
    ----------
    x, y:
        Bottom-left corner of the bar, in NDC.
    width, height:
        Full size of the bar, in NDC units.
    frac:
        Fraction of the bar to fill, from the bottom. Clamped to
        ``[0, 1]``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(outline, fill)``, each a shape ``(4, 2)`` array of rectangle
        corners ordered for ``GL_LINE_LOOP``/``GL_TRIANGLE_FAN``. ``outline``
        spans the full bar; ``fill`` spans the filled portion.
    """
    frac = max(0.0, min(frac, 1.0))

    outline = np.array([
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height],
    ])

    fill_height = height * frac
    fill = np.array([
        [x, y],
        [x + width, y],
        [x + width, y + fill_height],
        [x, y + fill_height],
    ])

    return outline, fill


def axis_tick_vertices(z_min: float, z_max: float, n_ticks: int,
                       x_ndc: float) -> tuple[np.ndarray, list[float]]:
    """Generate tick mark positions along the altitude axis.

    Parameters
    ----------
    z_min, z_max:
        Altitude view bounds [m].
    n_ticks:
        Number of evenly spaced tick marks to generate, including both
        endpoints.
    x_ndc:
        X coordinate at which to place the tick marks, in NDC.

    Returns
    -------
    tuple[np.ndarray, list[float]]
        Shape ``(n_ticks, 2)`` array of ``(x, y)`` NDC tick positions, and a
        list of the corresponding altitude values [m].
    """
    altitudes = np.linspace(z_min, z_max, n_ticks)
    verts = np.array([
        [x_ndc, altitude_to_ndc_y(z, z_min, z_max)] for z in altitudes
    ])
    return verts, altitudes.tolist()
