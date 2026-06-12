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

from viz_colormap import temperature_to_color
from viz_geometry import (
    arrow_vertices,
    flame_vertices,
    ground_line_vertices,
    progress_bar_vertices,
    segmented_rocket_body_vertices,
)
from viz_input import EnvControlState
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
        ``"flame"``, ``"temp_cold"``, ``"temp_hot"``, ``"text"``,
        ``"weight_arrow"``, ``"thrust_arrow"``, ``"net_arrow"``,
        ``"fuel_bar_bg"``, ``"fuel_bar_fill"``) to ``(r, g, b)`` tuples in
        ``[0, 1]``.
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
                    z_min: float, z_max: float, temps: dict[str, float],
                    thermal_cfg: dict) -> None:
        """Draw the segmented rocket body and its exhaust flame.

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
        temps:
            Per-segment temperatures [K], keyed by ``"forward"``, ``"mid"``,
            ``"aft"`` (see :class:`thermal.ThermalModel`).
        thermal_cfg:
            ``config.yaml``'s ``thermal`` section, used for the
            ``t_min``/``t_max`` color-gradient range.
        """
        center_y = altitude_to_ndc_y(z, z_min, z_max)
        forward, mid, aft = segmented_rocket_body_vertices(
            ROCKET_CENTER_X, center_y, self.rocket_width, self.rocket_height)

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

        t_min, t_max = thermal_cfg["t_min"], thermal_cfg["t_max"]
        cold, hot = self.colors["temp_cold"], self.colors["temp_hot"]
        for poly, seg_name in ((forward, "forward"), (mid, "mid"), (aft, "aft")):
            r, g, b = temperature_to_color(temps[seg_name], t_min, t_max, cold, hot)
            glColor3f(r, g, b)
            glBegin(GL_TRIANGLE_FAN)
            for x, y in poly:
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

    def _draw_segment_labels(self, center_y: float, temps: dict[str, float],
                              thermal_cfg: dict) -> None:
        """Draw small "FWD/MID/AFT NNNK" labels beside the rocket body."""
        hud = self.hud_cfg
        label_x = ROCKET_CENTER_X + self.rocket_width / 2.0 + hud["segment_label_offset_x"]

        half_h = self.rocket_height / 2.0
        nose_h = min(self.rocket_width, self.rocket_height)
        shoulder_y = center_y + half_h - nose_h
        body_bottom = center_y - half_h
        seg_h = max(shoulder_y - body_bottom, 1e-6) / 3.0

        positions = {
            "aft": body_bottom + seg_h * 0.5,
            "mid": body_bottom + seg_h * 1.5,
            "forward": body_bottom + seg_h * 2.5,
        }
        labels = {"forward": "FWD", "mid": "MID", "aft": "AFT"}

        t_min, t_max = thermal_cfg["t_min"], thermal_cfg["t_max"]
        cold, hot = self.colors["temp_cold"], self.colors["temp_hot"]
        text_h = hud["segment_label_text_height"]
        for seg, y in positions.items():
            color = temperature_to_color(temps[seg], t_min, t_max, cold, hot)
            self.text_renderer.draw_text(
                f"{labels[seg]} {temps[seg]:.0f}K", label_x, y, text_h,
                color, self.screen_width, self.screen_height)

    def _draw_env_panel(self, env_state: EnvControlState) -> None:
        """Draw the top-right ambient temperature/humidity readout."""
        hud = self.hud_cfg
        margin = hud["margin"]
        text_h = hud["text_height"]
        line_step = text_h * 1.4
        x0 = 1.0 - margin - hud["env_panel_width"]
        y0 = 1.0 - margin

        temp_str = f"{env_state.ambient_temp:.0f}K"
        hum_str = f"{env_state.humidity:.0f}%"
        if env_state.mode == "temp":
            temp_str = f"[{env_state.buffer or '_'}]"
        if env_state.mode == "humidity":
            hum_str = f"[{env_state.buffer or '_'}]"

        self._draw_text(f"TEMP: {temp_str}", x0, y0)
        self._draw_text(f"HUM:  {hum_str}", x0, y0 - line_step)

    def draw_hud(self, diag: dict, center_y: float, temps: dict[str, float],
                  thermal_cfg: dict, env_state: EnvControlState) -> None:
        """Draw the diagnostics overlay (HUD).

        Parameters
        ----------
        diag:
            Diagnostic values as returned by
            :func:`viz_diagnostics.compute_diagnostics`.
        center_y:
            Rocket body's vertical center in NDC, as returned by
            :func:`viz_transform.altitude_to_ndc_y` for the current
            altitude -- used to position the segment temperature labels.
        temps:
            Per-segment temperatures [K], keyed by ``"forward"``, ``"mid"``,
            ``"aft"``.
        thermal_cfg:
            ``config.yaml``'s ``thermal`` section, used for the
            ``t_min``/``t_max`` color-gradient range.
        env_state:
            Live ambient temperature/humidity control state.
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

        # --- Segment temperature labels and environment panel ------------
        self._draw_segment_labels(center_y, temps, thermal_cfg)
        self._draw_env_panel(env_state)
