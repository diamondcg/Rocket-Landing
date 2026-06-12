"""Color-gradient helpers for the simulation visualization.

Pure math only -- no SDL/OpenGL dependencies, so this module is safe to
import and unit test without a display or GL context.
"""

from __future__ import annotations


def temperature_to_color(temp: float, t_min: float, t_max: float,
                          cold_color: tuple[float, float, float],
                          hot_color: tuple[float, float, float]
                          ) -> tuple[float, float, float]:
    """Map a temperature to an RGB color via linear cold->hot interpolation.

    Parameters
    ----------
    temp:
        Temperature value [K] (or any consistent unit).
    t_min, t_max:
        Temperature range mapped to ``cold_color``..``hot_color``.
        ``t_max`` must be greater than ``t_min``.
    cold_color, hot_color:
        ``(r, g, b)`` triples in ``[0, 1]`` for the cold and hot ends of
        the gradient.

    Returns
    -------
    tuple[float, float, float]
        Interpolated ``(r, g, b)``. ``temp <= t_min`` maps to
        ``cold_color`` and ``temp >= t_max`` maps to ``hot_color``.
    """
    frac = (temp - t_min) / (t_max - t_min)
    frac = max(0.0, min(frac, 1.0))
    return tuple(c + frac * (h - c) for c, h in zip(cold_color, hot_color))
