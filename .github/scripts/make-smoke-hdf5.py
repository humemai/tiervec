#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "h5py>=3.10"]
# ///
"""make-smoke-hdf5.py OUT.hdf5 ROWS DIM QUERIES: a small Euclidean set in the layout of the ann-benchmarks files, for the CI smoke only.

Integer coordinates from 0 to 255 around a few centers, like SIFT. The neighbors are the 10 nearest rows by an exact scan.
"""
import sys

import h5py
import numpy as np


def main() -> None:
    out, rows, dim, queries = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rng = np.random.default_rng(7)
    centers = rng.integers(0, 256, size=(32, dim))

    def draw(n: int) -> np.ndarray:
        pick = rng.integers(0, len(centers), size=n)
        return np.clip(centers[pick] + rng.normal(0, 40, size=(n, dim)), 0, 255).round().astype(np.float32)

    train, test = draw(rows), draw(queries)
    neighbors = np.empty((queries, 10), dtype=np.int32)
    for i, q in enumerate(test):
        d = ((train.astype(np.float64) - q.astype(np.float64)) ** 2).sum(axis=1)
        neighbors[i] = np.lexsort((np.arange(rows), d))[:10]
    with h5py.File(out, "w") as f:
        f.attrs["distance"] = "euclidean"
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)
        f.create_dataset("neighbors", data=neighbors)
    print(f"wrote {out}: {rows} rows, {dim} dimensions, {queries} queries")


if __name__ == "__main__":
    main()
