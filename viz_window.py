"""SDL2 window and OpenGL context management for the visualization.

This module owns all SDL imports/calls. It is intentionally thin and is not
unit tested, since it requires a display and GL context to run.
"""

from __future__ import annotations

import ctypes

import sdl2
from OpenGL.GL import glViewport

_KEY_MAP = {
    sdl2.SDLK_t: "t",
    sdl2.SDLK_h: "h",
    sdl2.SDLK_PLUS: "+",
    sdl2.SDLK_KP_PLUS: "+",
    sdl2.SDLK_EQUALS: "+",
    sdl2.SDLK_MINUS: "-",
    sdl2.SDLK_KP_MINUS: "-",
    sdl2.SDLK_RETURN: "enter",
    sdl2.SDLK_KP_ENTER: "enter",
    sdl2.SDLK_ESCAPE: "escape",
    sdl2.SDLK_BACKSPACE: "backspace",
}
for _digit in range(10):
    _KEY_MAP[getattr(sdl2, f"SDLK_{_digit}")] = str(_digit)
    _KEY_MAP[getattr(sdl2, f"SDLK_KP_{_digit}")] = str(_digit)


class GLWindow:
    """An SDL2 window with an OpenGL context.

    Parameters
    ----------
    width, height:
        Window dimensions [px].
    title:
        Window title.
    vsync:
        If ``True``, enable vsync via ``SDL_GL_SetSwapInterval``.
    """

    def __init__(self, width: int, height: int, title: str,
                 vsync: bool = True) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.vsync = vsync
        self._window = None
        self._gl_context = None

    def __enter__(self) -> "GLWindow":
        sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)

        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MAJOR_VERSION, 2)
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MINOR_VERSION, 1)

        self._window = sdl2.SDL_CreateWindow(
            self.title.encode("utf-8"),
            sdl2.SDL_WINDOWPOS_CENTERED,
            sdl2.SDL_WINDOWPOS_CENTERED,
            self.width,
            self.height,
            sdl2.SDL_WINDOW_OPENGL,
        )
        self._gl_context = sdl2.SDL_GL_CreateContext(self._window)
        sdl2.SDL_GL_SetSwapInterval(1 if self.vsync else 0)

        glViewport(0, 0, self.width, self.height)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._gl_context is not None:
            sdl2.SDL_GL_DeleteContext(self._gl_context)
        if self._window is not None:
            sdl2.SDL_DestroyWindow(self._window)
        sdl2.SDL_Quit()

    def poll_events(self) -> tuple[bool, list[str]]:
        """Process the SDL event queue.

        Returns
        -------
        tuple[bool, list[str]]
            ``(should_continue, key_events)``. ``should_continue`` is
            ``False`` only on ``SDL_QUIT`` (window close) -- Esc no longer
            unconditionally quits here; callers must dispatch ``'escape'``
            themselves based on application state. ``key_events`` is the
            ordered list of key-name strings (``'t'``, ``'h'``, ``'+'``,
            ``'-'``, ``'0'``-``'9'``, ``'enter'``, ``'escape'``,
            ``'backspace'``) for keydown events this frame. Unmapped keys
            are ignored.
        """
        running = True
        keys: list[str] = []
        event = sdl2.SDL_Event()
        while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == sdl2.SDL_QUIT:
                running = False
            elif event.type == sdl2.SDL_KEYDOWN:
                name = _KEY_MAP.get(event.key.keysym.sym)
                if name is not None:
                    keys.append(name)
        return running, keys

    def swap(self) -> None:
        """Present the rendered frame."""
        sdl2.SDL_GL_SwapWindow(self._window)
