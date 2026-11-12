#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "pyarrow>=15"]
# ///
"""Write a small parquet data set for the CI smoke of the benchmark harness (docs/BENCHMARKS.md, "The CI smoke").

It has the layout of the shards of the DBpedia OpenAI set that scripts/prep_real_dataset.py reads: several files, a string column ``_id`` and
a column of lists of doubles named ``text-embedding-3-large-1536-embedding``. The vectors are clustered Gaussian noise, and their lengths are
not 1 (the prep script has to scale them). Nothing is committed: the job writes it into target/, runs the prep script on it, and runs the
harness on the result.

    uv run --python 3.12 .github/scripts/make-smoke-parquet.py OUT_DIR ROWS DIM SHARDS
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

COLUMN = "text-embedding-3-large-1536-embedding"


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    rows, dim, shards = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    centres = rng.normal(size=(20, dim))
    x = centres[rng.integers(0, 20, size=rows)] + 0.5 * rng.normal(size=(rows, dim))
    x *= rng.uniform(0.5, 2.0, size=(rows, 1))
    for i, part in enumerate(np.array_split(np.arange(rows), shards)):
        table = pa.table({
            "_id": pa.array([f"<row/{j}>" for j in part], type=pa.string()),
            COLUMN: pa.array([x[j].tolist() for j in part], type=pa.list_(pa.float64())),
        })
        pq.write_table(table, out / f"train-{i:05d}-of-{shards:05d}.parquet", row_group_size=500)
    print(f"wrote {rows} rows of dimension {dim} in {shards} files to {out}")


if __name__ == "__main__":
    main()
