#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "h5py>=3.10"]
# ///
"""Convert an ann-benchmarks HDF5 file with Euclidean distance (such as ``sift-128-euclidean.hdf5``) into the files of the benchmark harness.

Run it with uv, which installs the two dependencies named in the header above::

    uv run --python 3.12 scripts/prep_ann_hdf5.py sift-128-euclidean.hdf5 OUT_DIR

The HDF5 file holds ``train`` (the index), ``test`` (the queries) and ``neighbors`` (for each query, the rows of ``train`` that are nearest by the
Euclidean distance), and the attribute ``distance=euclidean``. ``OUT_DIR`` gets the four files of format 1 that ``prep_real_dataset.py`` writes
(``vectors.f32``, ``queries.f32``, ``truth.i32``, ``meta.properties``), with ``metric=l2`` and the vectors as they are, not normalised. The
benchmark programs that take an l2 set say so in docs/BENCHMARKS.md.

The script takes the first ``--queries`` test vectors and the first ``--truth-k`` neighbors of each. It checks the neighbors of the first
``--check`` queries against its own exact scan of ``train`` (a returned row must be no farther than the k-th nearest), and refuses the file
when they differ. ``meta.properties`` is written last, and a directory without it is refused by the harness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

VECTORS, QUERIES, TRUTH, META = "vectors.f32", "queries.f32", "truth.i32", "meta.properties"
CHUNK = 100_000


def distances(train: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances of every row of train to q, in float64."""
    d = train.astype(np.float64) - q.astype(np.float64)
    return np.einsum("ij,ij->i", d, d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="the HDF5 file")
    ap.add_argument("out", type=Path, help="the output directory (created)")
    ap.add_argument("--queries", type=int, default=1000, help="the first N test vectors (default: %(default)s)")
    ap.add_argument("--truth-k", type=int, default=10, help="neighbors per query (default: %(default)s)")
    ap.add_argument("--check", type=int, default=20, help="queries checked against an exact scan (default: %(default)s)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing data set in the output directory")
    args = ap.parse_args()

    out: Path = args.out
    if (out / META).exists() and not args.force:
        raise SystemExit(f"{out / META} exists; pass --force to rebuild it")
    out.mkdir(parents=True, exist_ok=True)
    (out / META).unlink(missing_ok=True)

    with h5py.File(args.src, "r") as f:
        distance = f.attrs.get("distance")
        if isinstance(distance, bytes):
            distance = distance.decode()
        if distance != "euclidean":
            raise SystemExit(f"{args.src.name}: distance is {distance!r}; this script takes Euclidean sets only")
        train = f["train"]
        count, dim = train.shape
        test = f["test"]
        neighbors = f["neighbors"]
        if test.shape[1] != dim:
            raise SystemExit(f"the queries have dimension {test.shape[1]}, the index {dim}")
        if not 1 <= args.queries <= test.shape[0] or neighbors.shape[0] != test.shape[0] or not 1 <= args.truth_k <= neighbors.shape[1]:
            raise SystemExit(f"--queries {args.queries} or --truth-k {args.truth_k} do not fit {test.shape[0]} queries with {neighbors.shape[1]} neighbors")
        tmp = out / (VECTORS + ".tmp")
        with open(tmp, "wb") as fv:
            for start in range(0, count, CHUNK):
                block = np.asarray(train[start:start + CHUNK], dtype="<f4")
                if not np.isfinite(block).all():
                    raise SystemExit(f"rows from {start} hold a value that is not a finite number")
                block.tofile(fv)
        q = np.asarray(test[:args.queries], dtype="<f4")
        truth = np.asarray(neighbors[:args.queries, :args.truth_k], dtype="<i4")
    if truth.min() < 0 or truth.max() >= count:
        raise SystemExit("a neighbor is not a row of train")

    mm = np.memmap(tmp, dtype="<f4", mode="r", shape=(count, dim))
    for i in range(min(args.check, args.queries)):
        d = np.concatenate([distances(mm[s:s + CHUNK], q[i]) for s in range(0, count, CHUNK)])
        kth = np.partition(d, args.truth_k - 1)[args.truth_k - 1]
        if len(set(truth[i].tolist())) != args.truth_k or (d[truth[i]] > kth * (1 + 1e-6)).any():
            raise SystemExit(f"the neighbors of query {i} are not the {args.truth_k} nearest rows")
    del mm

    q.tofile(out / QUERIES)
    truth.tofile(out / TRUTH)
    tmp.replace(out / VECTORS)
    meta = {
        "format": 1,
        "dim": dim,
        "count": count,
        "queries": args.queries,
        "truth-k": args.truth_k,
        "metric": "l2",
        "normalized": "false",
        "source": args.src.name,
        "heldout": "test-set-of-the-file",
        "checked-queries": min(args.check, args.queries),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prep-script": "scripts/prep_ann_hdf5.py",
    }
    with open(out / META, "w", encoding="utf-8") as f:
        for key, value in meta.items():
            f.write(f"{key}={value}\n")
    print(f"done: {count} vectors of dimension {dim}, {args.queries} queries with {args.truth_k} neighbors each, in {out}")


if __name__ == "__main__":
    main()
