"""Entry point for the rocket landing visualization.

Runs a single simulation via :func:`simulation.run_simulation`, then
replays the recorded log in an SDL2/OpenGL window.
"""

from __future__ import annotations

import time

from simulation import load_config, run_simulation
from viz_diagnostics import compute_diagnostics
from viz_playback import PlaybackClock
from viz_renderer import SceneRenderer
from viz_transform import compute_view_bounds
from viz_window import GLWindow


def main(config_path: str = "config.yaml", seed: int | None = 42) -> None:
    """Run a simulation and play it back in an SDL2/OpenGL window.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.
    seed:
        Random seed passed to :func:`simulation.run_simulation`.
    """
    cfg = load_config(config_path)
    result = run_simulation(cfg, seed=seed)
    viz_cfg = cfg["visualization"]
    rocket_cfg = cfg["rocket"]
    dt = cfg["simulation"]["dt"]

    z = result["z"]
    v = result["v"]
    mass = result["mass"]
    thrust = result["thrust"]
    time_log = result["time"]
    n_samples = len(z)

    z_min, z_max = compute_view_bounds(z, viz_cfg["view_padding_frac"])

    colors = {name: tuple(rgb) for name, rgb in viz_cfg["colors"].items()}
    renderer = SceneRenderer(
        colors=colors,
        rocket_size=(viz_cfg["rocket_width"], viz_cfg["rocket_height"]),
        hud_cfg=viz_cfg["hud"],
        screen_size=(viz_cfg["window_width"], viz_cfg["window_height"]),
    )
    clock = PlaybackClock(
        dt=dt,
        n_samples=n_samples,
        playback_speed=viz_cfg["playback_speed"],
        loop=viz_cfg["loop"],
    )

    frame_period = 1.0 / viz_cfg["fps"]
    thrust_max = rocket_cfg["thrust_max"]

    with GLWindow(viz_cfg["window_width"], viz_cfg["window_height"],
                   viz_cfg["window_title"]) as window:
        last_time = time.perf_counter()
        running = True
        while running:
            now = time.perf_counter()
            real_dt = now - last_time
            last_time = now

            running = window.poll_events()

            index = clock.advance(real_dt)

            renderer.clear()
            renderer.draw_ground(z_min, z_max)
            renderer.draw_rocket(z[index], thrust[index], thrust_max,
                                  z_min, z_max)
            if viz_cfg["hud"]["enabled"]:
                diag = compute_diagnostics(
                    z[index], v[index], mass[index], thrust[index],
                    time_log[index], rocket_cfg["gravity"],
                    rocket_cfg["mass_dry"], rocket_cfg["mass_init"],
                    thrust_max)
                renderer.draw_hud(diag)
            window.swap()

            if clock.finished:
                running = False

            elapsed = time.perf_counter() - now
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)


if __name__ == "__main__":
    main()
