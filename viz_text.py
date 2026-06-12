"""SDL_ttf-based text rendering for the simulation visualization HUD.

This module owns all SDL_ttf and texture-upload calls. It is intentionally
thin and is not unit tested, since it requires SDL_ttf and a GL context to
run.
"""

from __future__ import annotations

import ctypes

import sdl2
import sdl2.sdlttf as ttf
from OpenGL.GL import (
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_LINEAR,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_QUADS,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glColor4f,
    glDeleteTextures,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex2f,
)


class TextRenderer:
    """Renders strings to OpenGL textures using SDL_ttf.

    Parameters
    ----------
    font_path:
        Path to a TrueType/OpenType font file.
    font_size_px:
        Font size used to rasterize glyphs [px].
    """

    def __init__(self, font_path: str, font_size_px: int) -> None:
        ttf.TTF_Init()
        self._font = ttf.TTF_OpenFont(font_path.encode("utf-8"), font_size_px)
        if not self._font:
            raise RuntimeError(
                f"Failed to load font {font_path!r}: "
                f"{ttf.TTF_GetError().decode('utf-8')}"
            )

    def render_to_texture(self, text: str,
                           color: tuple[float, float, float]
                           ) -> tuple[int, int, int]:
        """Render text to a new OpenGL texture.

        Parameters
        ----------
        text:
            String to render. Must be non-empty.
        color:
            ``(r, g, b)`` glyph color in ``[0, 1]``.

        Returns
        -------
        tuple[int, int, int]
            ``(texture_id, width_px, height_px)``. The caller is responsible
            for deleting the texture with ``glDeleteTextures``.
        """
        r, g, b = (int(c * 255) for c in color)
        sdl_color = sdl2.SDL_Color(r, g, b, 255)

        surface = ttf.TTF_RenderUTF8_Blended(self._font, text.encode("utf-8"),
                                              sdl_color)
        converted = sdl2.SDL_ConvertSurfaceFormat(
            surface, sdl2.SDL_PIXELFORMAT_ABGR8888, 0)
        sdl2.SDL_FreeSurface(surface)

        surf = converted.contents
        width, height = surf.w, surf.h
        pixels = ctypes.string_at(surf.pixels, surf.pitch * height)

        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, pixels)

        sdl2.SDL_FreeSurface(converted)

        return texture_id, width, height

    def draw_text(self, text: str, x_ndc: float, y_ndc: float,
                   height_ndc: float, color: tuple[float, float, float],
                   screen_width: int, screen_height: int) -> None:
        """Draw a string as a textured quad.

        Parameters
        ----------
        text:
            String to render. Empty strings are skipped.
        x_ndc, y_ndc:
            Top-left corner of the text, in NDC.
        height_ndc:
            Height of the rendered text, in NDC. Width is derived from the
            glyph texture's aspect ratio and the window's pixel aspect
            ratio.
        color:
            ``(r, g, b)`` glyph color in ``[0, 1]``.
        screen_width, screen_height:
            Window dimensions [px], used to preserve the glyph aspect ratio.
        """
        if not text:
            return

        texture_id, width_px, height_px = self.render_to_texture(text, color)

        width_ndc = height_ndc * (width_px / height_px) * (
            screen_height / screen_width)

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(x_ndc, y_ndc)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(x_ndc + width_ndc, y_ndc)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(x_ndc + width_ndc, y_ndc - height_ndc)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(x_ndc, y_ndc - height_ndc)
        glEnd()

        glDisable(GL_BLEND)
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures(1, [texture_id])

    def close(self) -> None:
        """Release the loaded font and shut down SDL_ttf."""
        if self._font:
            ttf.TTF_CloseFont(self._font)
            self._font = None
        ttf.TTF_Quit()

    def __del__(self) -> None:
        self.close()
