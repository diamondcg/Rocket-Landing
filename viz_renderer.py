"""OpenGL drawing for the simulation visualization.

This module owns all OpenGL drawing calls. It is intentionally thin --
vertex data is computed by :mod:`viz_geometry` and :mod:`viz_transform` --
and is not unit tested, since it requires a GL context to run.
"""

from __future__ import annotations

from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_LINE_LOOP,
    GL_LINES,
    GL_QUADS,
    GL_TRIANGLE_FAN,
    glBegin,
    glClear,
    glClearColor,
    glColor3f,
    glEnd,
    glLineWidth,
    glVertex2f,
)

from viz_geometry import (
    arrow_vertices,
    flame_vertices,
    ground_line_vertices,
    progress_bar_vertices,
    rocket_body_vertices,
)
from viz_text import TextRenderer
from viz_transform import altitude_to_ndc_y

ROCKET_CENTER_X = 0.0
ARROW_LINE_WIDTH_PX = 3.0


class SceneRenderer:
    """Draws one simulation frame using OpenGL primitives.

    Parameters
    ----------
    colors:
        Mapping of element name (``"background"``, ``"ground"``,
        ``"rocket"``, ``"flame"``, ``"text"``, ``"weight_arrow"``,
        ``"thrust_arrow"``, ``"net_arrow"``, ``"fuel_bar_bg"``,
        ``"fuel_bar_fill"``) to ``(r, g, b)`` tuples in ``[0, 1]``.
    rocket_size:
        ``(width, height)`` of the rocket body, in NDC units.
    hud_cfg:
        ``visualization.hud`` configuration dict (font, layout, scaling).
    screen_size:
        ``(width, height)`` of the window, in pixels.
    """

    def __init__(self, colors: dict[str, tuple[float, float, float]],
                 rocket_size: tuple[float, float], hud_cfg: dict,
                 screen_size: tuple[int, int]) -> None:
        self.colors = colors
        self.rocket_width, self.rocket_height = rocket_size
        self.hud_cfg = hud_cfg
        self.screen_width, self.screen_height = screen_size
        self.text_renderer = TextRenderer(hud_cfg["font_path"],
                                           hud_cfg["font_size_px"])

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

    def _draw_text(self, text: str, x_ndc: float, y_ndc: float) -> None:
        """Draw a HUD text string at the given top-left NDC position."""
        self.text_renderer.draw_text(
            text, x_ndc, y_ndc, self.hud_cfg["text_height"],
            self.colors["text"], self.screen_width, self.screen_height)

    def _draw_arrow(self, origin_x: float, origin_y: float, magnitude: float,
                     direction: int, color: tuple[float, float, float]
                     ) -> None:
        """Draw a vertical force arrow at the given NDC origin."""
        hud = self.hud_cfg
        shaft, head = arrow_vertices(
            origin_x, origin_y, magnitude, hud["arrow_max_force"],
            hud["arrow_max_length"], direction, hud["arrow_shaft_width"])

        r, g, b = color
        glColor3f(r, g, b)

        glLineWidth(ARROW_LINE_WIDTH_PX)
        glBegin(GL_LINES)
        for x, y in shaft:
            glVertex2f(x, y)
        glEnd()

        glBegin(GL_TRIANGLE_FAN)
        for x, y in head:
            glVertex2f(x, y)
        glEnd()

    def draw_hud(self, diag: dict) -> None:
        """Draw the diagnostics overlay (HUD).

        Parameters
        ----------
        diag:
            Diagnostic values as returned by
            :func:`viz_diagnostics.compute_diagnostics`.
        """
        hud = self.hud_cfg
        margin = hud["margin"]
        text_h = hud["text_height"]
        line_step = text_h * 1.4

        # --- Top-left telemetry panel ---------------------------------
        x0 = -1.0 + margin
        y0 = 1.0 - margin
        lines = [
            f"ALT  {diag['altitude']:7.1f} M",
            f"VEL  {diag['velocity']:7.1f} M/S",
            f"MASS {diag['mass']:7.1f} KG",
            f"TIME {diag['time']:7.1f} S",
        ]
        for i, line in enumerate(lines):
            self._draw_text(line, x0, y0 - i * line_step)

        # --- Bottom-left force diagram ---------------------------------
        label_gap = 0.02
        label_y = -1.0 + margin + text_h
        baseline_y = label_y + label_gap + hud["arrow_max_length"]
        arrow_spacing = 0.15
        arrow_x0 = -1.0 + margin + hud["arrow_shaft_width"] * 3.0

        self._draw_arrow(arrow_x0, baseline_y, diag["weight"], -1,
                          self.colors["weight_arrow"])
        self._draw_text(f"W {diag['weight']:.0f}N",
                        arrow_x0 - hud["arrow_shaft_width"] * 3.0, label_y)

        thrust_x = arrow_x0 + arrow_spacing
        self._draw_arrow(thrust_x, baseline_y, diag["thrust_force"], 1,
                          self.colors["thrust_arrow"])
        self._draw_text(f"T {diag['thrust_force']:.0f}N",
                        thrust_x - hud["arrow_shaft_width"] * 3.0, label_y)

        net_force = diag["net_force"]
        net_x = arrow_x0 + 2 * arrow_spacing
        net_direction = 1 if net_force >= 0 else -1
        self._draw_arrow(net_x, baseline_y, abs(net_force), net_direction,
                          self.colors["net_arrow"])
        self._draw_text(f"N {net_force:.0f}N",
                        net_x - hud["arrow_shaft_width"] * 3.0, label_y)

        # --- Right-edge fuel gauge ---------------------------------------
        bar_w = hud["fuel_bar_width"]
        bar_h = hud["fuel_bar_height"]
        bar_x = 1.0 - margin - bar_w
        bar_y = -1.0 + margin
        outline, fill = progress_bar_vertices(bar_x, bar_y, bar_w, bar_h,
                                               diag["fuel_frac"])

        r, g, b = self.colors["fuel_bar_fill"]
        glColor3f(r, g, b)
        glBegin(GL_QUADS)
        for x, y in fill:
            glVertex2f(x, y)
        glEnd()

        r, g, b = self.colors["fuel_bar_bg"]
        glColor3f(r, g, b)
        glBegin(GL_LINE_LOOP)
        for x, y in outline:
            glVertex2f(x, y)
        glEnd()

        self._draw_text(f"{diag['fuel_frac'] * 100:.0f}%", bar_x - 0.01,
                        bar_y + bar_h + line_step * 2)
        self._draw_text("FUEL", bar_x - 0.01, bar_y + bar_h + line_step)
