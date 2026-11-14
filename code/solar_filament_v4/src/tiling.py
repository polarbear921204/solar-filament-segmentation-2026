from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int


def _starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_tiles(width: int, height: int, tile_size: int = 1024, stride: int = 768) -> list[Tile]:
    xs = _starts(width, tile_size, stride)
    ys = _starts(height, tile_size, stride)
    return [Tile(x0=x, y0=y, x1=min(x + tile_size, width), y1=min(y + tile_size, height)) for y in ys for x in xs]
