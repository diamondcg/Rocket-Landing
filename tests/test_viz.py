"""Unit tests for the rocket landing visualization modules.

Only imports modules that are importable without a display/GL context
(``viz_transform``, ``viz_geometry``, ``viz_playback``, ``viz_diagnostics``);
``viz_window``, ``viz_renderer``, ``viz_text``, and ``visualize`` require
SDL/SDL_ttf/OpenGL and are not unit tested here.
"""

import sys
import os

import numpy as np
import pytest

# Ensure the project root is on the path so imports work when tests are run
# from any directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation import load_config
from viz_diagnostics import compute_diagnostics
from viz_transform import (
    altitude_to_ndc_y,
    altitude_to_screen_y,
    compute_view_bounds,
    world_to_ndc,
)
from viz_geometry import (
    arrow_vertices,
    axis_tick_vertices,
    flame_vertices,
    ground_line_vertices,
    progress_bar_vertices,
    rocket_body_vertices,
)
from viz_playback import (
    PlaybackClock,
    index_for_time,
    is_playback_finished,
    loop_index_for_time,
)


def _default_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    return load_config(cfg_path)


# ===========================================================================
# viz_transform.py
# ===========================================================================

class TestVizTransform:
    def test_altitude_to_ndc_endpoints(self):
        assert np.isclose(altitude_to_ndc_y(0, 0, 100), -1.0)
        assert np.isclose(altitude_to_ndc_y(100, 0, 100), 1.0)

    def test_altitude_to_ndc_midpoint(self):
        assert np.isclose(altitude_to_ndc_y(50, 0, 100), 0.0)

    def test_altitude_to_screen_y_inverted_axis(self):
        top = altitude_to_screen_y(100, 0, 100, 600)
        bottom = altitude_to_screen_y(0, 0, 100, 600)
        assert top < bottom

    def test_altitude_to_screen_y_endpoints(self):
        assert np.isclose(altitude_to_screen_y(100, 0, 100, 600), 0.0)
        assert np.isclose(altitude_to_screen_y(0, 0, 100, 600), 600.0)

    def test_altitude_to_screen_y_margin(self):
        top = altitude_to_screen_y(100, 0, 100, 600, margin_px=50)
        bottom = altitude_to_screen_y(0, 0, 100, 600, margin_px=50)
        assert np.isclose(top, 50.0)
        assert np.isclose(bottom, 550.0)

    def test_world_to_ndc_range(self):
        x, y = world_to_ndc(50, 50, (0, 100), (0, 100))
        assert np.isclose(x, 0.0)
        assert np.isclose(y, 0.0)

    def test_world_to_ndc_endpoints(self):
        x, y = world_to_ndc(0, 100, (0, 100), (0, 100))
        assert np.isclose(x, -1.0)
        assert np.isclose(y, 1.0)

    def test_compute_view_bounds_includes_ground(self):
        z = np.array([100.0, 80.0, 50.0, 10.0, 0.0])
        z_min, z_max = compute_view_bounds(z, padding_frac=0.1)
        assert z_min <= 0.0
        assert z_max > z.max()

    def test_compute_view_bounds_no_padding(self):
        z = np.array([10.0, 5.0, 0.0])
        z_min, z_max = compute_view_bounds(z, padding_frac=0.0)
        assert np.isclose(z_max, 10.0)
        assert np.isclose(z_min, 0.0)

    def test_compute_view_bounds_handles_all_positive(self):
        z = np.array([100.0, 90.0, 80.0])
        z_min, z_max = compute_view_bounds(z, padding_frac=0.1)
        assert z_min <= 0.0
        assert z_max >= 100.0


# ===========================================================================
# viz_geometry.py
# ===========================================================================

class TestVizGeometry:
    def test_rocket_body_vertices_shape(self):
        verts = rocket_body_vertices(0.0, 0.0, 0.1, 0.2)
        assert verts.shape == (5, 2)

    def test_rocket_body_vertices_centered_horizontally(self):
        verts = rocket_body_vertices(0.0, 0.0, 0.1, 0.2)
        assert np.isclose(verts[:, 0].min(), -0.05)
        assert np.isclose(verts[:, 0].max(), 0.05)

    def test_rocket_body_vertices_spans_height(self):
        verts = rocket_body_vertices(0.0, 0.0, 0.1, 0.2)
        assert np.isclose(verts[:, 1].max(), 0.1)
        assert np.isclose(verts[:, 1].min(), -0.1)

    def test_rocket_body_vertices_offset_center(self):
        verts = rocket_body_vertices(0.5, 0.5, 0.1, 0.2)
        assert np.isclose(verts[:, 0].mean(), 0.5, atol=0.05)

    def test_flame_vertices_shape(self):
        verts = flame_vertices(0.0, 0.0, thrust_frac=0.5, width=0.05,
                                max_height=0.3)
        assert verts.shape == (3, 2)

    def test_flame_vertices_scale_with_thrust(self):
        low = flame_vertices(0.0, 0.0, thrust_frac=0.1, width=0.05,
                              max_height=0.3)
        high = flame_vertices(0.0, 0.0, thrust_frac=1.0, width=0.05,
                               max_height=0.3)
        low_extent = abs(low[:, 1].min())
        high_extent = abs(high[:, 1].min())
        assert high_extent > low_extent

    def test_flame_vertices_zero_thrust(self):
        verts = flame_vertices(0.0, 0.0, thrust_frac=0.0, width=0.05,
                                max_height=0.3)
        assert np.allclose(verts[:, 1], 0.0)

    def test_flame_vertices_full_thrust_length(self):
        verts = flame_vertices(0.0, 0.0, thrust_frac=1.0, width=0.05,
                                max_height=0.3)
        assert np.isclose(verts[:, 1].min(), -0.3)

    def test_ground_line_vertices_shape(self):
        verts = ground_line_vertices(y_ndc=-0.5)
        assert verts.shape == (2, 2)
        assert np.allclose(verts[:, 1], -0.5)

    def test_ground_line_vertices_extent(self):
        verts = ground_line_vertices(y_ndc=0.0, x_extent=0.8)
        assert np.isclose(verts[0, 0], -0.8)
        assert np.isclose(verts[1, 0], 0.8)

    def test_axis_tick_vertices_count(self):
        verts, labels = axis_tick_vertices(0, 100, n_ticks=5, x_ndc=-0.9)
        assert verts.shape == (5, 2)
        assert len(labels) == 5

    def test_axis_tick_vertices_label_range(self):
        _, labels = axis_tick_vertices(0, 100, n_ticks=5, x_ndc=-0.9)
        assert np.isclose(labels[0], 0.0)
        assert np.isclose(labels[-1], 100.0)

    def test_axis_tick_vertices_x_position(self):
        verts, _ = axis_tick_vertices(0, 100, n_ticks=3, x_ndc=-0.9)
        assert np.allclose(verts[:, 0], -0.9)

    def test_arrow_vertices_shapes(self):
        shaft, head = arrow_vertices(0.0, 0.0, magnitude=100.0,
                                      max_magnitude=200.0, max_length=0.2,
                                      direction=1, shaft_width=0.01)
        assert shaft.shape == (2, 2)
        assert head.shape == (3, 2)

    def test_arrow_vertices_length_scales_with_magnitude(self):
        small, _ = arrow_vertices(0.0, 0.0, magnitude=50.0,
                                   max_magnitude=200.0, max_length=0.2,
                                   direction=1, shaft_width=0.01)
        large, _ = arrow_vertices(0.0, 0.0, magnitude=150.0,
                                   max_magnitude=200.0, max_length=0.2,
                                   direction=1, shaft_width=0.01)
        small_len = abs(small[1, 1] - small[0, 1])
        large_len = abs(large[1, 1] - large[0, 1])
        assert large_len > small_len

    def test_arrow_vertices_clamps_at_max_magnitude(self):
        at_max, _ = arrow_vertices(0.0, 0.0, magnitude=200.0,
                                    max_magnitude=200.0, max_length=0.2,
                                    direction=1, shaft_width=0.01)
        over_max, _ = arrow_vertices(0.0, 0.0, magnitude=1000.0,
                                      max_magnitude=200.0, max_length=0.2,
                                      direction=1, shaft_width=0.01)
        assert np.isclose(at_max[1, 1], over_max[1, 1])
        assert np.isclose(at_max[1, 1], 0.2)

    def test_arrow_vertices_direction(self):
        up, _ = arrow_vertices(0.0, 0.0, magnitude=100.0,
                                max_magnitude=200.0, max_length=0.2,
                                direction=1, shaft_width=0.01)
        down, _ = arrow_vertices(0.0, 0.0, magnitude=100.0,
                                  max_magnitude=200.0, max_length=0.2,
                                  direction=-1, shaft_width=0.01)
        assert up[1, 1] > 0.0
        assert down[1, 1] < 0.0
        assert np.isclose(up[1, 1], -down[1, 1])

    def test_arrow_vertices_zero_magnitude(self):
        shaft, _ = arrow_vertices(0.0, 0.0, magnitude=0.0,
                                   max_magnitude=200.0, max_length=0.2,
                                   direction=1, shaft_width=0.01)
        assert np.allclose(shaft[0], shaft[1])

    def test_progress_bar_vertices_shapes(self):
        outline, fill = progress_bar_vertices(0.0, 0.0, 0.1, 0.5, frac=0.5)
        assert outline.shape == (4, 2)
        assert fill.shape == (4, 2)

    def test_progress_bar_vertices_full(self):
        outline, fill = progress_bar_vertices(0.0, 0.0, 0.1, 0.5, frac=1.0)
        assert np.isclose(fill[:, 1].max(), outline[:, 1].max())

    def test_progress_bar_vertices_empty(self):
        _, fill = progress_bar_vertices(0.0, 0.0, 0.1, 0.5, frac=0.0)
        assert np.allclose(fill[:, 1], 0.0)

    def test_progress_bar_vertices_clamps_frac(self):
        _, fill_over = progress_bar_vertices(0.0, 0.0, 0.1, 0.5, frac=2.0)
        _, fill_under = progress_bar_vertices(0.0, 0.0, 0.1, 0.5, frac=-1.0)
        assert np.isclose(fill_over[:, 1].max(), 0.5)
        assert np.allclose(fill_under[:, 1], 0.0)

    def test_progress_bar_vertices_position(self):
        outline, _ = progress_bar_vertices(0.2, 0.3, 0.1, 0.5, frac=0.5)
        assert np.isclose(outline[:, 0].min(), 0.2)
        assert np.isclose(outline[:, 1].min(), 0.3)


# ===========================================================================
# viz_diagnostics.py
# ===========================================================================

class TestVizDiagnostics:
    def _compute(self, **overrides):
        defaults = dict(
            z=50.0, v=-5.0, mass=15.0, thrust=100.0, t=2.0,
            gravity=9.81, mass_dry=10.0, mass_init=20.0, thrust_max=500.0,
        )
        defaults.update(overrides)
        return compute_diagnostics(**defaults)

    def test_weight_equals_mass_times_gravity(self):
        diag = self._compute(mass=15.0, gravity=9.81)
        assert np.isclose(diag["weight"], 15.0 * 9.81)

    def test_net_force_is_thrust_minus_weight(self):
        diag = self._compute(mass=15.0, thrust=100.0, gravity=9.81)
        assert np.isclose(diag["net_force"], 100.0 - 15.0 * 9.81)

    def test_speed_is_absolute_velocity(self):
        diag = self._compute(v=-7.5)
        assert diag["speed"] == 7.5
        assert diag["velocity"] == -7.5

    def test_fuel_frac_midpoint(self):
        # mass_dry=10, mass_init=20 -> fuel capacity=10; mass=15 -> 5/10
        diag = self._compute(mass=15.0, mass_dry=10.0, mass_init=20.0)
        assert np.isclose(diag["fuel_frac"], 0.5)

    def test_fuel_frac_clamped_below_dry_mass(self):
        diag = self._compute(mass=5.0, mass_dry=10.0, mass_init=20.0)
        assert diag["fuel_frac"] == 0.0

    def test_fuel_frac_clamped_above_init_mass(self):
        diag = self._compute(mass=25.0, mass_dry=10.0, mass_init=20.0)
        assert diag["fuel_frac"] == 1.0

    def test_pass_through_values(self):
        diag = self._compute(z=42.0, t=3.5, mass=12.0, thrust=80.0,
                              thrust_max=500.0)
        assert diag["altitude"] == 42.0
        assert diag["time"] == 3.5
        assert diag["mass"] == 12.0
        assert diag["thrust_force"] == 80.0
        assert diag["thrust_max"] == 500.0


# ===========================================================================
# viz_playback.py
# ===========================================================================

class TestVizPlayback:
    def test_index_for_time_start(self):
        assert index_for_time(0.0, dt=0.01, n_samples=1000) == 0

    def test_index_for_time_progresses(self):
        idx1 = index_for_time(0.1, dt=0.01, n_samples=1000)
        idx2 = index_for_time(0.2, dt=0.01, n_samples=1000)
        assert idx1 == 10
        assert idx2 == 20

    def test_index_for_time_clamped(self):
        idx = index_for_time(1000.0, dt=0.01, n_samples=100)
        assert idx == 99

    def test_index_for_time_playback_speed(self):
        idx_normal = index_for_time(0.1, dt=0.01, n_samples=1000,
                                     playback_speed=1.0)
        idx_fast = index_for_time(0.1, dt=0.01, n_samples=1000,
                                   playback_speed=2.0)
        assert idx_fast == 2 * idx_normal

    def test_is_playback_finished_false_at_start(self):
        assert not is_playback_finished(0.0, dt=0.01, n_samples=100)

    def test_is_playback_finished_true_past_end(self):
        assert is_playback_finished(2.0, dt=0.01, n_samples=100)

    def test_is_playback_finished_at_exact_end(self):
        assert is_playback_finished(1.0, dt=0.01, n_samples=100)

    def test_loop_index_wraps(self):
        n = 100
        dt = 0.01
        total = n * dt
        idx = loop_index_for_time(total + 0.01, dt=dt, n_samples=n)
        assert idx == 1

    def test_loop_index_in_range(self):
        idx = loop_index_for_time(5.0, dt=0.01, n_samples=100)
        assert 0 <= idx < 100

    def test_playback_clock_advance(self):
        clock = PlaybackClock(dt=0.01, n_samples=100)
        idx0 = clock.advance(0.0)
        idx1 = clock.advance(0.05)
        assert idx0 == 0
        assert idx1 == 5

    def test_playback_clock_finished_flag(self):
        clock = PlaybackClock(dt=0.01, n_samples=10)
        clock.advance(1.0)
        assert clock.finished

    def test_playback_clock_not_finished_initially(self):
        clock = PlaybackClock(dt=0.01, n_samples=10)
        assert not clock.finished

    def test_playback_clock_loop_never_finishes(self):
        clock = PlaybackClock(dt=0.01, n_samples=10, loop=True)
        clock.advance(0.2)  # exceeds 10 * 0.01 = 0.1s
        assert not clock.finished

    def test_playback_clock_loop_wraps_index(self):
        clock = PlaybackClock(dt=0.01, n_samples=10, loop=True)
        idx = clock.advance(0.105)
        assert 0 <= idx < 10

    def test_playback_clock_reset(self):
        clock = PlaybackClock(dt=0.01, n_samples=10)
        clock.advance(1.0)
        clock.reset()
        assert not clock.finished
        assert clock.advance(0.0) == 0


# ===========================================================================
# config.yaml visualization section
# ===========================================================================

class TestVizConfig:
    def test_visualization_section_present(self):
        cfg = _default_cfg()
        assert "visualization" in cfg

    def test_visualization_required_keys(self):
        cfg = _default_cfg()
        viz = cfg["visualization"]
        for key in ("window_width", "window_height", "window_title", "fps",
                    "playback_speed", "loop", "view_padding_frac",
                    "rocket_width", "rocket_height", "colors", "hud"):
            assert key in viz

    def test_visualization_colors_are_rgb_triples(self):
        cfg = _default_cfg()
        for name, rgb in cfg["visualization"]["colors"].items():
            assert len(rgb) == 3
            assert all(0.0 <= c <= 1.0 for c in rgb)

    def test_visualization_window_dimensions_positive(self):
        cfg = _default_cfg()
        viz = cfg["visualization"]
        assert viz["window_width"] > 0
        assert viz["window_height"] > 0
        assert viz["fps"] > 0

    def test_hud_required_keys(self):
        cfg = _default_cfg()
        hud = cfg["visualization"]["hud"]
        for key in ("enabled", "font_path", "font_size_px", "text_height",
                    "margin", "arrow_max_force", "arrow_max_length",
                    "arrow_shaft_width", "fuel_bar_width", "fuel_bar_height"):
            assert key in hud

    def test_hud_font_path_exists(self):
        cfg = _default_cfg()
        font_path = cfg["visualization"]["hud"]["font_path"]
        assert os.path.exists(font_path)

    def test_hud_colors_present_and_valid(self):
        cfg = _default_cfg()
        colors = cfg["visualization"]["colors"]
        for name in ("text", "weight_arrow", "thrust_arrow", "net_arrow",
                      "fuel_bar_bg", "fuel_bar_fill"):
            assert name in colors
            rgb = colors[name]
            assert len(rgb) == 3
            assert all(0.0 <= c <= 1.0 for c in rgb)
