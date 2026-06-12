"""SDL2 window and OpenGL context management for the visualization.

This module owns all SDL imports/calls. It is intentionally thin and is not
unit tested, since it requires a display and GL context to run.
"""

from __future__ import annotations

import ctypes

import sdl2
from OpenGL.GL import glViewport


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

    def poll_events(self) -> bool:
        """Process the SDL event queue.

        Returns
        -------
        bool
            ``False`` if a quit event was received (window closed or Esc
            pressed), ``True`` otherwise.
        """
        event = sdl2.SDL_Event()
        while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == sdl2.SDL_QUIT:
                return False
            if event.type == sdl2.SDL_KEYDOWN:
                if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                    return False
        return True

    def swap(self) -> None:
        """Present the rendered frame."""
        sdl2.SDL_GL_SwapWindow(self._window)
