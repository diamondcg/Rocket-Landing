"""Unit tests for the rocket landing visualization modules.

Only imports modules that are importable without a display/GL context
(``viz_transform``, ``viz_geometry``, ``viz_playback``); ``viz_window``,
``viz_renderer``, and ``visualize`` require SDL/OpenGL and are not unit
tested here.
"""

import sys
import os

import numpy as np
import pytest

# Ensure the project root is on the path so imports work when tests are run
# from any directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation import load_config
from viz_transform import (
    altitude_to_ndc_y,
    altitude_to_screen_y,
    compute_view_bounds,
    world_to_ndc,
)
from viz_geometry import (
    axis_tick_vertices,
    flame_vertices,
    ground_line_vertices,
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
                    "rocket_width", "rocket_height", "colors"):
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
