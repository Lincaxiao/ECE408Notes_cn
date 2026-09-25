#!/usr/bin/env python3
"""CPU verification of lecture formulas and Strategy-2 indexing.

No CUDA compiler or GPU is used. Python's float arithmetic checks the mapping,
not the rounding behavior or runtime correctness of CUDA execution.
"""
from __future__ import annotations

import math
import random
from collections import Counter


def convolution_naive(image: list[list[float]], mask: list[list[float]]) -> list[list[float]]:
    height, width, m = len(image), len(image[0]), len(mask)
    radius = m // 2
    output = [[0.0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            for i in range(m):
                for j in range(m):
                    yy, xx = y - radius + i, x - radius + j
                    value = image[yy][xx] if 0 <= yy < height and 0 <= xx < width else 0.0
                    output[y][x] += mask[i][j] * value
    return output


def convolution_tiled(image: list[list[float]], mask: list[list[float]], tile_width: int) -> list[list[float]]:
    height, width, m = len(image), len(image[0]), len(mask)
    radius, side = m // 2, tile_width + m - 1
    output = [[0.0] * width for _ in range(height)]
    writes: Counter[tuple[int, int]] = Counter()
    for y0 in range(0, height, tile_width):
        for x0 in range(0, width, tile_width):
            tile = [[0.0] * side for _ in range(side)]
            # Emulate completion of every loading thread before computation.
            for ty in range(side):
                for tx in range(side):
                    yi, xi = y0 + ty - radius, x0 + tx - radius
                    if 0 <= yi < height and 0 <= xi < width:
                        tile[ty][tx] = image[yi][xi]
            for ty in range(tile_width):
                for tx in range(tile_width):
                    value = 0.0
                    for i in range(m):
                        for j in range(m):
                            assert ty + i < side and tx + j < side
                            value += mask[i][j] * tile[ty + i][tx + j]
                    yo, xo = y0 + ty, x0 + tx
                    if yo < height and xo < width:
                        output[yo][xo] = value
                        writes[yo, xo] += 1
    assert len(writes) == height * width and set(writes.values()) == {1}
    return output


def direct_counts(height: int, width: int, t: int, m: int, y0: int, x0: int) -> tuple[int, int]:
    """Count actual in-range input reads; ghosts do not read global memory."""
    radius, side = m // 2, t + m - 1
    reads = 0
    for y in range(y0, min(height, y0 + t)):
        for x in range(x0, min(width, x0 + t)):
            for i in range(m):
                for j in range(m):
                    if 0 <= y - radius + i < height and 0 <= x - radius + j < width:
                        reads += 1
    loads = sum(0 <= y0 + ty - radius < height and 0 <= x0 + tx - radius < width
                for ty in range(side) for tx in range(side))
    return reads, loads


def count_axis(n: int, start: int, t: int, m: int) -> tuple[int, int]:
    radius = m // 2
    accesses = sum(max(0, min(n - 1, x + radius) - max(0, x - radius) + 1)
                   for x in range(start, min(n, start + t)))
    loads = max(0, min(n - 1, start + t + radius - 1) - max(0, start - radius) + 1)
    return accesses, loads


def main() -> None:
    rng = random.Random(40809)
    cases = [(32, 32, 16, 5), (37, 53, 16, 5), (1, 1, 8, 5),
             (3, 7, 8, 9), (19, 25, 8, 3), (64, 64, 8, 5),
             (5, 9, 3, 1), (2, 1, 1, 3)]
    block_count = 0
    for height, width, t, m in cases:
        image = [[rng.uniform(-2.0, 2.0) for _ in range(width)] for _ in range(height)]
        # An asymmetric mask also verifies the unflipped indexing convention.
        mask = [[rng.uniform(-1.0, 1.0) for _ in range(m)] for _ in range(m)]
        expected, actual = convolution_naive(image, mask), convolution_tiled(image, mask, t)
        error = max(abs(expected[y][x] - actual[y][x]) for y in range(height) for x in range(width))
        assert error == 0.0, (height, width, t, m, error)
        for y0 in range(0, height, t):
            for x0 in range(0, width, t):
                ay, by = count_axis(height, y0, t, m)
                ax, bx = count_axis(width, x0, t, m)
                assert direct_counts(height, width, t, m, y0, x0) == (ay * ax, by * bx)
                block_count += 1
    print(f"PASS: {len(cases)} CPU image/mask cases; each output written once.")
    print(f"PASS: {block_count} tiles; direct enumeration equals boundary-product formulas.")

    for t in (1, 2, 3, 8, 16, 32):
        for m in (1, 3, 5, 9):
            uses = Counter(out + tap for out in range(t) for tap in range(m))
            assert len(uses) == t + m - 1 and sum(uses.values()) == t * m
    assert [Counter(out + tap for out in range(8) for tap in range(5))[x]
            for x in range(12)] == [1, 2, 3, 4, 5, 5, 5, 5, 4, 3, 2, 1]
    assert count_axis(32, 0, 8, 5) == (37, 10)
    assert direct_counts(32, 32, 16, 5, 0, 0) == (5929, 324)
    print("PASS: 1D multiplicities, 37/10 boundary ratio, 5929/324 corner ratio.")

    divergent, inactive = [], []
    for warp in range(32):
        role = [(thread % 32 < 30 and thread // 32 < 30)
                for thread in range(32 * warp, 32 * (warp + 1))]
        if any(role) and not all(role):
            divergent.append(warp)
        if not any(role):
            inactive.append(warp)
    assert divergent == list(range(30)) and inactive == [30, 31]
    print("PASS: 30 computation-divergent warps, 2 uniformly inactive warps.")

    intensity, byte_ratio = 37400 / 696, 696 / 37400
    reuse = 2 * 37400 / 696
    assert math.isclose(intensity * byte_ratio, 1.0)
    assert math.isclose(reuse, 2 * intensity)
    print(f"PASS: A40 example b={byte_ratio:.7f} B/FLOP, I={intensity:.4f} FLOP/B, R={reuse:.4f}.")
    reuse80 = (80 * 7 / (80 + 7 - 1)) ** 2
    print(f"PASS: T=80,m=7 model: R={reuse80:.6f}; peak fraction={0.0192 * reuse80:.6%}.")
    print("LIMIT: these checks do not compile or execute CUDA and do not measure GPU performance.")


if __name__ == "__main__":
    main()
