"""OpenGL drawing for the simulation visualization.

This module owns all OpenGL drawing calls. It is intentionally thin --
vertex data is computed by :mod:`viz_geometry` and :mod:`viz_transform` --
and is not unit tested, since it requires a GL context to run.
"""

from __future__ import annotations

from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_LINES,
    GL_TRIANGLE_FAN,
    glBegin,
    glClear,
    glClearColor,
    glColor3f,
    glEnd,
    glVertex2f,
)

from viz_geometry import flame_vertices, ground_line_vertices, rocket_body_vertices
from viz_transform import altitude_to_ndc_y

ROCKET_CENTER_X = 0.0


class SceneRenderer:
    """Draws one simulation frame using OpenGL primitives.

    Parameters
    ----------
    colors:
        Mapping of element name (``"background"``, ``"ground"``,
        ``"rocket"``, ``"flame"``) to ``(r, g, b)`` tuples in ``[0, 1]``.
    rocket_size:
        ``(width, height)`` of the rocket body, in NDC units.
    """

    def __init__(self, colors: dict[str, tuple[float, float, float]],
                 rocket_size: tuple[float, float]) -> None:
        self.colors = colors
        self.rocket_width, self.rocket_height = rocket_size

    def clear(self) -> None:
        """Clear the frame with the configured background color."""
        r, g, b = self.colors["background"]
        glClearColor(r, g, b, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

    def draw_ground(self, z_min: float, z_max: float) -> None:
        """Draw the ground line at ``z = 0``.

        Parameters
        ----------
        z_min, z_max:
            Altitude view bounds [m].
        """
        y_ndc = altitude_to_ndc_y(0.0, z_min, z_max)
        verts = ground_line_vertices(y_ndc)

        r, g, b = self.colors["ground"]
        glColor3f(r, g, b)
        glBegin(GL_LINES)
        for x, y in verts:
            glVertex2f(x, y)
        glEnd()

    def draw_rocket(self, z: float, thrust: float, thrust_max: float,
                    z_min: float, z_max: float) -> None:
        """Draw the rocket body and its exhaust flame.

        Parameters
        ----------
        z:
            Current altitude [m].
        thrust:
            Current thrust command [N].
        thrust_max:
            Maximum thrust [N], used to normalize the flame length.
        z_min, z_max:
            Altitude view bounds [m].
        """
        center_y = altitude_to_ndc_y(z, z_min, z_max)
        body = rocket_body_vertices(ROCKET_CENTER_X, center_y,
                                     self.rocket_width, self.rocket_height)

        thrust_frac = thrust / thrust_max if thrust_max > 0 else 0.0
        flame_base_y = center_y - self.rocket_height / 2.0
        flame = flame_vertices(ROCKET_CENTER_X, flame_base_y, thrust_frac,
                                self.rocket_width, self.rocket_height)

        r, g, b = self.colors["flame"]
        glColor3f(r, g, b)
        glBegin(GL_TRIANGLE_FAN)
        for x, y in flame:
            glVertex2f(x, y)
        glEnd()

        r, g, b = self.colors["rocket"]
        glColor3f(r, g, b)
        glBegin(GL_TRIANGLE_FAN)
        for x, y in body:
            glVertex2f(x, y)
        glEnd()
