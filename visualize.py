"""Entry point for the rocket landing visualization.

Runs a single simulation via :func:`simulation.run_simulation`, then
replays the recorded log in an SDL2/OpenGL window.
"""

from __future__ import annotations

import time

from simulation import load_config, run_simulation
from thermal import ThermalModel
from viz_diagnostics import compute_diagnostics
from viz_input import EnvControlState
from viz_playback import PlaybackClock
from viz_renderer import SceneRenderer
from viz_transform import altitude_to_ndc_y, compute_view_bounds
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
    thermal_cfg = cfg["thermal"]
    env_cfg = cfg["environment"]
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

    thermal_model = ThermalModel(
        segment_params=thermal_cfg["segment_params"],
        initial_temps=thermal_cfg["initial_temps"],
    )
    env_state = EnvControlState(
        ambient_temp=env_cfg["ambient_temp_init"],
        humidity=env_cfg["humidity_init"],
        temp_step=env_cfg["temp_step"],
        humidity_step=env_cfg["humidity_step"],
        temp_min=env_cfg["temp_min"], temp_max=env_cfg["temp_max"],
        humidity_min=env_cfg["humidity_min"], humidity_max=env_cfg["humidity_max"],
    )

    frame_period = 1.0 / viz_cfg["fps"]
    thrust_max = rocket_cfg["thrust_max"]

    with GLWindow(viz_cfg["window_width"], viz_cfg["window_height"],
                   viz_cfg["window_title"]) as window:
        last_time = time.perf_counter()
        running = True
        previous_index = 0
        while running:
            now = time.perf_counter()
            real_dt = now - last_time
            last_time = now

            running, keys = window.poll_events()
            for key in keys:
                if key == "escape" and env_state.mode is None:
                    running = False
                else:
                    env_state.handle_key(key)

            index = clock.advance(real_dt)
            if clock.loop and index < previous_index:
                thermal_model.reset()
            previous_index = index

            temps = thermal_model.step(
                v[index], thrust[index], thrust_max,
                env_state.ambient_temp, env_state.humidity, dt)

            renderer.clear()
            renderer.draw_ground(z_min, z_max)
            renderer.draw_rocket(z[index], thrust[index], thrust_max,
                                  z_min, z_max, temps, thermal_cfg)
            if viz_cfg["hud"]["enabled"]:
                diag = compute_diagnostics(
                    z[index], v[index], mass[index], thrust[index],
                    time_log[index], rocket_cfg["gravity"],
                    rocket_cfg["mass_dry"], rocket_cfg["mass_init"],
                    thrust_max)
                center_y = altitude_to_ndc_y(z[index], z_min, z_max)
                renderer.draw_hud(diag, center_y, temps, thermal_cfg, env_state)
            window.swap()

            if clock.finished:
                running = False

            elapsed = time.perf_counter() - now
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)


if __name__ == "__main__":
    main()
