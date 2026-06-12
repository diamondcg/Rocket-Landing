"""Playback timing helpers for the simulation visualization.

These functions map elapsed wall-clock time to an index into a recorded
simulation log (as returned by ``simulation.run_simulation``). Pure
stdlib/numpy only -- safe to import and unit test without a display or GL
context.
"""

from __future__ import annotations


def index_for_time(elapsed_s: float, dt: float, n_samples: int,
                    playback_speed: float = 1.0) -> int:
    """Compute the recorded-sample index for an elapsed wall time.

    Parameters
    ----------
    elapsed_s:
        Wall-clock seconds since playback start.
    dt:
        Simulation time step [s] (``cfg["simulation"]["dt"]``).
    n_samples:
        Length of the recorded arrays.
    playback_speed:
        Playback speed multiplier; ``> 1`` plays faster than real time.

    Returns
    -------
    int
        Sample index in ``[0, n_samples - 1]``, clamped once the elapsed
        time exceeds the recording length.
    """
    sim_time = elapsed_s * playback_speed
    index = int(sim_time / dt)
    return max(0, min(index, n_samples - 1))


def is_playback_finished(elapsed_s: float, dt: float, n_samples: int,
                          playback_speed: float = 1.0) -> bool:
    """Return whether playback has reached the end of the recording.

    Parameters
    ----------
    elapsed_s:
        Wall-clock seconds since playback start.
    dt:
        Simulation time step [s].
    n_samples:
        Length of the recorded arrays.
    playback_speed:
        Playback speed multiplier.

    Returns
    -------
    bool
        ``True`` once the (speed-scaled) elapsed time exceeds the
        recording's total duration (``n_samples * dt``).
    """
    sim_time = elapsed_s * playback_speed
    total_duration = n_samples * dt
    return sim_time >= total_duration


def loop_index_for_time(elapsed_s: float, dt: float, n_samples: int,
                         playback_speed: float = 1.0) -> int:
    """Compute a looping recorded-sample index for an elapsed wall time.

    Like :func:`index_for_time`, but wraps around (modulo the recording
    duration) instead of clamping.

    Parameters
    ----------
    elapsed_s:
        Wall-clock seconds since playback start.
    dt:
        Simulation time step [s].
    n_samples:
        Length of the recorded arrays.
    playback_speed:
        Playback speed multiplier.

    Returns
    -------
    int
        Sample index in ``[0, n_samples - 1]``.
    """
    sim_time = elapsed_s * playback_speed
    total_duration = n_samples * dt
    wrapped = sim_time % total_duration
    index = int(wrapped / dt)
    return max(0, min(index, n_samples - 1))


class PlaybackClock:
    """Tracks elapsed playback time across frame ticks.

    Wraps :func:`index_for_time`, :func:`loop_index_for_time`, and
    :func:`is_playback_finished` as a stateful convenience for a render
    loop; all timing math is delegated to those functions.

    Parameters
    ----------
    dt:
        Simulation time step [s].
    n_samples:
        Length of the recorded arrays.
    playback_speed:
        Playback speed multiplier.
    loop:
        If ``True``, playback restarts from the beginning once it reaches
        the end of the recording instead of stopping.
    """

    def __init__(self, dt: float, n_samples: int,
                 playback_speed: float = 1.0, loop: bool = False) -> None:
        self.dt = dt
        self.n_samples = n_samples
        self.playback_speed = playback_speed
        self.loop = loop
        self._elapsed_s = 0.0

    def advance(self, real_dt_s: float) -> int:
        """Advance elapsed time and return the current sample index.

        Parameters
        ----------
        real_dt_s:
            Wall-clock time elapsed since the previous call [s].

        Returns
        -------
        int
            Current sample index in ``[0, n_samples - 1]``.
        """
        self._elapsed_s += real_dt_s
        if self.loop:
            return loop_index_for_time(self._elapsed_s, self.dt,
                                        self.n_samples, self.playback_speed)
        return index_for_time(self._elapsed_s, self.dt, self.n_samples,
                               self.playback_speed)

    def reset(self) -> None:
        """Reset elapsed playback time to zero."""
        self._elapsed_s = 0.0

    @property
    def finished(self) -> bool:
        """``True`` if playback has reached the end and is not looping."""
        if self.loop:
            return False
        return is_playback_finished(self._elapsed_s, self.dt, self.n_samples,
                                      self.playback_speed)
