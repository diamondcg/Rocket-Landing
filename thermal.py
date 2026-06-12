"""Illustrative per-segment thermal model for the visualization.

This is a parallel, viz-time-only model -- it is NOT part of the
estimator/controller state ``[z, v, mass]`` and does not feed back into
``dynamics.py`` or ``simulation.py``. It is advanced once per render frame
during playback, using the replayed velocity/thrust at the current
playback index plus the live ambient temperature and humidity from
:class:`viz_input.EnvControlState`.

All temperatures (segment temperatures and ``ambient_temp``) are in the
same unit (Kelvin by convention, per ``config.yaml``'s ``thermal`` and
``environment`` sections) so the convective cooling term needs no unit
conversion.

Extension points (for future, more aerospace-accurate models):
  - ``segment_params`` entries are looked up by key, so new per-segment
    coefficients (e.g. radiative or solar heating) can be added without
    changing :meth:`ThermalModel.step`'s signature.
  - :meth:`ThermalModel.step` can be extended with optional keyword
    arguments (e.g. altitude/air density) that default to preserving the
    current behaviour, allowing a gradual transition to a higher-fidelity
    model.
"""

from __future__ import annotations

SEGMENTS = ("forward", "mid", "aft")


class ThermalModel:
    """Tracks illustrative forward/mid/aft fuselage temperatures.

    Parameters
    ----------
    segment_params:
        Mapping keyed by ``"forward"``, ``"mid"``, ``"aft"``, each a dict
        with:

        - ``aero_coeff``: aerodynamic heating rate, applied as
          ``aero_coeff * v**2`` [K/s per (m/s)^2].
        - ``engine_coeff``: engine heating rate, applied as
          ``engine_coeff * (thrust / thrust_max)`` [K/s].
        - ``conv_coeff``: convective cooling rate toward ``ambient_temp``
          [1/s], scaled by a humidity factor.
    initial_temps:
        Mapping keyed by ``"forward"``, ``"mid"``, ``"aft"`` -> initial
        segment temperature [K].
    """

    def __init__(self, segment_params: dict[str, dict[str, float]],
                 initial_temps: dict[str, float]) -> None:
        self.segment_params = segment_params
        self._initial_temps = dict(initial_temps)
        self.temps = dict(initial_temps)

    def reset(self) -> None:
        """Reset all segment temperatures to their initial values."""
        self.temps = dict(self._initial_temps)

    def step(self, v: float, thrust: float, thrust_max: float,
             ambient_temp: float, humidity: float, dt: float
             ) -> dict[str, float]:
        """Advance segment temperatures by ``dt`` and return the new state.

        Parameters
        ----------
        v:
            Vertical velocity [m/s]. Only its magnitude matters (heating
            from aerodynamic effects is direction-agnostic).
        thrust:
            Current thrust command [N].
        thrust_max:
            Maximum thrust [N], used to normalize engine heating.
        ambient_temp:
            Ambient environmental temperature [K].
        humidity:
            Relative humidity [%], in ``[0, 100]``. Higher humidity
            increases the convective cooling rate toward ``ambient_temp``.
        dt:
            Time step [s].

        Returns
        -------
        dict[str, float]
            A copy of the updated segment temperatures, keyed by
            ``"forward"``, ``"mid"``, ``"aft"``.
        """
        humidity_factor = 1.0 + (humidity / 100.0)
        thrust_frac = thrust / thrust_max if thrust_max > 0 else 0.0

        for seg in SEGMENTS:
            params = self.segment_params[seg]
            aero_heat = params["aero_coeff"] * v ** 2
            engine_heat = params["engine_coeff"] * thrust_frac
            conv_cool = (params["conv_coeff"] * humidity_factor
                          * (self.temps[seg] - ambient_temp))
            self.temps[seg] += (aero_heat + engine_heat - conv_cool) * dt

        return dict(self.temps)
