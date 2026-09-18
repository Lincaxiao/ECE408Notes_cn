#!/usr/bin/env python3
"""CPU simulation of the lecture's CUDA indexing; no GPU or CUDA is used.

Each cooperative load phase is completed before the output phase, matching
an explicit block barrier. All shared slots are checked for initialization.
"""
from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Sequence


def reference(n: Sequence[int], m: Sequence[int]) -> list[int]:
    r, w = len(m) // 2, len(n)
    return [sum(n[g] * m[j] for j in range(len(m))
                if 0 <= (g := i - r + j) < w) for i in range(w)]


def simulated(n: Sequence[int], m: Sequence[int], b: int,
              method: str) -> list[int]:
    if b <= 0 or not m or len(m) % 2 != 1:
        raise ValueError("Positive block size and odd, nonempty mask required")
    r, k, w = len(m) // 2, len(m), len(n)
    if method == "three" and r > b:
        raise ValueError("Three-part loader requires r <= B")
    if method == "two" and 2 * r > b:
        raise ValueError("Two-pass loader requires 2r <= B")
    if method not in {"three", "two", "loop", "core"}:
        raise ValueError("Unknown method")
    out: list[int | None] = [None] * w

    def read(g: int) -> int:
        return n[g] if 0 <= g < w else 0

    for s in range(0, w, b):
        shared: list[int | None] = [None] * (b if method == "core" else b + 2*r)
        writes = [0] * len(shared)

        def put(q: int, value: int) -> None:
            assert 0 <= q < len(shared)
            writes[q] += 1
            shared[q] = value

        for t in range(b):
            if method == "three":
                if t >= b-r:
                    put(t-(b-r), read(s-b+t))
                put(r+t, read(s+t))
                if t < r:
                    put(r+b+t, read(s+b+t))
            elif method == "two":
                put(t, read(s+t-r))
                if t < 2*r:
                    put(t+b, read(s+t-r+b))
            elif method == "loop":
                for q in range(t, b+2*r, b):
                    put(q, read(s-r+q))
            else:
                put(t, read(s+t))
        assert all(x == 1 for x in writes), (method, b, k, s, writes)
        assert all(x is not None for x in shared)
        # Cooperative loading has completed: model the block barrier here.
        for t in range(b):
            value = 0
            for j in range(k):
                g = s+t-r+j
                if method == "core":
                    if not 0 <= g < w:
                        continue
                    q = t-r+j
                    v = shared[q] if s <= g < s+b else n[g]
                else:
                    q = t+j
                    assert 0 <= q < len(shared)
                    v = shared[q]
                assert v is not None
                value += v*m[j]
            if s+t < w:
                out[s+t] = value
    assert all(x is not None for x in out)
    return [int(x) for x in out if x is not None]


def main() -> None:
    rng = Random(40808)
    checks = 0
    widths = list(range(0, 50)) + [63, 64, 65, 127, 128, 129, 257]
    blocks = [1, 2, 3, 4, 5, 8, 16, 32, 128]
    for w in widths:
        n = [rng.randrange(-11, 12) for _ in range(w)]
        for k in [1, 3, 5, 7, 9, 13]:
            m = [rng.randrange(-7, 8) for _ in range(k)]
            ref = reference(n, m)
            r = k//2
            for b in blocks:
                for method in ["three", "two", "loop", "core"]:
                    if method == "three" and r > b:
                        continue
                    if method == "two" and 2*r > b:
                        continue
                    got = simulated(n, m, b, method)
                    assert got == ref, (w, b, k, method, got, ref)
                    checks += 1
    # Recheck all numerical examples reproduced from the lecture.
    n = list(range(1, 8)); m = [3, 4, 5, 4, 3]
    assert reference(n, m) == [22, 38, 57, 76, 95, 90, 74]
    mask2 = [[1,2,3,2,1],[2,3,4,3,2],[3,4,5,4,3],
             [2,3,4,3,2],[1,2,3,2,1]]
    patch = [[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],
             [4,5,6,7,8],[5,6,7,8,5]]
    assert sum(patch[i][j]*mask2[i][j] for i in range(5) for j in range(5)) == 321
    boundary = [[0,0,0,0,0],[0,0,1,2,3],[0,0,2,3,4],
                [0,0,3,4,5],[0,0,4,5,6]]
    assert sum(boundary[i][j]*mask2[i][j] for i in range(5) for j in range(5)) == 112
    ghost = [[0,0,0,0,0],[0,3,4,5,6],[0,2,3,4,5],
             [0,3,5,6,7],[0,1,1,3,1]]
    assert sum(ghost[i][j]*mask2[i][j] for i in range(5) for j in range(5)) == 179
    smooth = [[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],
              [4,16,26,16,4],[1,4,7,4,1]]
    assert sum(map(sum, smooth)) == 273
    valid = [sum(0 <= i-2+j < 7 for j in range(5)) for i in range(7)]
    assert valid == [3,4,5,5,5,4,3] and sum(valid) == 29
    assert [sum(0 <= s-2+q < 16 for q in range(8)) for s in [0,4,8,12]] == [6,8,8,6]
    result = {
        "status": "PASS", "cpu_indexing_comparisons": checks,
        "methods": ["three-part loader", "two-pass loader", "strided loader", "core-only cache"],
        "checks": ["numerical equality", "shared-memory bounds", "exactly-once shared initialization",
                   "first and last block", "partial blocks", "masks wider than input", "asymmetric weights"],
        "lecture_numeric_checks": {"1d": reference(n,m), "2d_interior": 321,
                                   "2d_boundary": 112, "2d_ghost": 179, "smoothing_weight_sum": 273},
        "gpu_execution": "NOT RUN", "timing_benchmark": "NOT RUN"
    }
    destination = Path(__file__).with_name("validation_results.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
