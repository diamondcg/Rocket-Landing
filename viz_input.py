"""Keyboard-driven environment control state for the visualization.

Pure stdlib only -- no SDL/OpenGL dependencies, so this module is safe to
import and unit test without a display or GL context.
"""

from __future__ import annotations


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


class EnvControlState:
    """Tracks live-adjustable ambient temperature and humidity.

    Parameters
    ----------
    ambient_temp:
        Initial ambient temperature [K].
    humidity:
        Initial relative humidity [%].
    temp_step:
        Amount ``ambient_temp`` changes per ``'+'``/``'-'`` keypress [K].
    humidity_step:
        Amount ``humidity`` changes per ``'+'``/``'-'`` keypress [%].
    temp_min, temp_max:
        Bounds for ``ambient_temp`` [K].
    humidity_min, humidity_max:
        Bounds for ``humidity`` [%].

    Notes
    -----
    Call :meth:`handle_key` once per relevant keydown event, in order. The
    expected key-name strings are ``'t'``, ``'h'``, ``'+'``, ``'-'``,
    ``'0'``-``'9'``, ``'enter'``, ``'escape'``, and ``'backspace'``.

    State machine
    -------------
    - When ``mode is None``: ``'t'`` enters temperature-edit mode and
      ``'h'`` enters humidity-edit mode (clearing ``buffer``); all other
      keys are ignored.
    - When a mode is active: ``'+'``/``'-'`` immediately adjust the active
      value by its configured step (clamped); digits append to ``buffer``;
      ``'backspace'`` removes the last buffered digit; ``'enter'`` parses a
      non-empty ``buffer`` as the new value (clamped) and exits the mode;
      ``'escape'`` exits the mode, discarding ``buffer`` but keeping any
      ``'+'``/``'-'`` adjustments already applied.
    """

    def __init__(self, ambient_temp: float, humidity: float,
                 temp_step: float, humidity_step: float,
                 temp_min: float, temp_max: float,
                 humidity_min: float = 0.0, humidity_max: float = 100.0
                 ) -> None:
        self.ambient_temp = ambient_temp
        self.humidity = humidity
        self.temp_step = temp_step
        self.humidity_step = humidity_step
        self.temp_min, self.temp_max = temp_min, temp_max
        self.humidity_min, self.humidity_max = humidity_min, humidity_max
        self.mode: str | None = None
        self.buffer: str = ""

    def handle_key(self, key: str) -> None:
        """Process one key-name string and update mode/buffer/values."""
        if self.mode is None:
            if key == "t":
                self.mode = "temp"
                self.buffer = ""
            elif key == "h":
                self.mode = "humidity"
                self.buffer = ""
            return

        if key == "+":
            self._adjust(+1)
        elif key == "-":
            self._adjust(-1)
        elif key in "0123456789":
            self.buffer += key
        elif key == "backspace":
            self.buffer = self.buffer[:-1]
        elif key == "enter":
            if self.buffer:
                self._apply(float(self.buffer))
            self.mode = None
            self.buffer = ""
        elif key == "escape":
            self.mode = None
            self.buffer = ""

    def _adjust(self, sign: int) -> None:
        if self.mode == "temp":
            self.ambient_temp = _clamp(
                self.ambient_temp + sign * self.temp_step,
                self.temp_min, self.temp_max)
        elif self.mode == "humidity":
            self.humidity = _clamp(
                self.humidity + sign * self.humidity_step,
                self.humidity_min, self.humidity_max)

    def _apply(self, value: float) -> None:
        if self.mode == "temp":
            self.ambient_temp = _clamp(value, self.temp_min, self.temp_max)
        elif self.mode == "humidity":
            self.humidity = _clamp(value, self.humidity_min, self.humidity_max)
