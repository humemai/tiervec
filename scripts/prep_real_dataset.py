#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "pyarrow>=15"]
# ///
"""Convert the parquet shards of an embedding set into the files that the benchmark harness reads (``Bench --data DIR``).

Run it with uv, which installs the two dependencies named in the header above::

    uv run --python 3.12 scripts/prep_real_dataset.py SHARDS_DIR OUT_DIR

``SHARDS_DIR`` holds ``*.parquet`` files (or is one such file). Each file has one column of embeddings, a list of floats per row (the default
column name is the one of the DBpedia OpenAI set with 1536 dimensions, ``text-embedding-3-large-1536-embedding``). ``OUT_DIR`` gets four files:

``vectors.f32``
    The index: ``count`` vectors of ``dim`` float32 values, little endian, row after row. The id of a vector is its row.
``queries.f32``
    ``--queries`` vectors held out of the set (drawn at random with ``--seed``). None of them is a row of ``vectors.f32``.
``truth.i32``
    For every query, the ``--truth-k`` rows of ``vectors.f32`` with the highest cosine, best first, as int32 little endian. They are exact: the
    script scans every vector (in blocks, so the memory stays bounded) and rescores the best candidates of each query in float64.
``meta.properties``
    ``key=value`` lines (``format``, ``dim``, ``count``, ``queries``, ``truth-k``, ``normalized`` and where the data came from). It is written
    last, so a directory without it is a run that did not finish, and the harness refuses it.

Every vector is scaled to length 1 (in float64, then stored as float32), so the dot product is the cosine and one number serves the tier and
JVector alike. The range of the lengths before scaling is kept in ``source-norm-min`` and ``source-norm-max``: for the OpenAI embeddings it is
1 to within float noise, so the scaling changes nothing visible. The harness refuses a file whose vectors are not unit length.

Memory: one batch of rows (``--batch-rows``) while reading, and one block of vectors plus a ``queries x block`` score matrix while scanning
(``--block-rows``, lowered when the queries are many). With the defaults and 1000 queries of dimension 1536 that is under 1 GiB, and the
script does not need more than 8 GiB for any setting that is sensible. Nothing is held for the whole set.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DEFAULT_COLUMN = "text-embedding-3-large-1536-embedding"
VECTORS, QUERIES, TRUTH, META = "vectors.f32", "queries.f32", "truth.i32", "meta.properties"
FORMAT = 1
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def list_shards(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    shards = sorted(src.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no *.parquet file in {src}")
    return shards


def matrix_of(column: pa.Array | pa.ChunkedArray, name: str) -> np.ndarray:
    """One batch of a list-of-numbers column as a float64 matrix of shape (rows, dim)."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if column.null_count:
        raise SystemExit(f"column {name} has {column.null_count} null embeddings")
    t = column.type
    if pa.types.is_fixed_size_list(t):
        dim = t.list_size
    elif pa.types.is_list(t) or pa.types.is_large_list(t):
        lengths = pc.min_max(pc.list_value_length(column)).as_py()
        if lengths["min"] != lengths["max"]:
            raise SystemExit(f"column {name} has embeddings of different lengths ({lengths['min']} to {lengths['max']})")
        dim = lengths["min"]
    else:
        raise SystemExit(f"column {name} has the type {t}, not a list of numbers")
    values = column.flatten().to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float64).reshape(len(column), dim)


def normalise(m64: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Scale every row to length 1 in float64 and store it as little endian float32; also the smallest and the largest length before."""
    norms = np.linalg.norm(m64, axis=1)
    if not np.all(norms > 0):
        raise SystemExit("a vector has length 0, which has no cosine")
    return (m64 / norms[:, None]).astype("<f4"), float(norms.min()), float(norms.max())


def read_and_split(shards: list[Path], column: str, out_dir: Path, queries: int, limit: int | None, seed: int, batch_rows: int):
    """Pass 1: stream the rows once. The held-out rows go to the query matrix, the others to vectors.f32.tmp (in the order they have)."""
    counts = [pq.ParquetFile(s).metadata.num_rows for s in shards]
    total = sum(counts)
    used = total if limit is None else min(total, limit)
    if queries < 1 or queries >= used:
        raise SystemExit(f"--queries {queries} needs fewer than the {used} rows that are used")
    held = np.sort(np.random.default_rng(seed).choice(used, size=queries, replace=False))
    log(f"{len(shards)} shard(s), {total} rows, using the first {used}, holding out {queries} queries")
    tmp = out_dir / (VECTORS + ".tmp")
    qmat = None
    dim = None
    qfilled = kept = g = 0
    nmin, nmax = float("inf"), 0.0
    with open(tmp, "wb") as fv:
        for si, shard in enumerate(shards):
            if g >= used:
                break
            pf = pq.ParquetFile(shard)
            if column not in pf.schema_arrow.names:
                raise SystemExit(f"{shard.name} has no column {column!r}; its columns are {pf.schema_arrow.names}")
            for batch in pf.iter_batches(batch_size=batch_rows, columns=[column]):
                if g >= used:
                    break
                m64 = matrix_of(batch.column(0), column)
                take = min(m64.shape[0], used - g)
                if take == 0:
                    continue
                m64 = m64[:take]
                if dim is None:
                    dim = m64.shape[1]
                    qmat = np.empty((queries, dim), dtype="<f4")
                elif m64.shape[1] != dim:
                    raise SystemExit(f"{shard.name} has vectors of dimension {m64.shape[1]}, the first rows had {dim}")
                m32, lo, hi = normalise(m64)
                nmin, nmax = min(nmin, lo), max(nmax, hi)
                a, b = np.searchsorted(held, [g, g + take])
                local = held[a:b] - g
                if local.size:
                    qmat[qfilled:qfilled + local.size] = m32[local]
                    qfilled += local.size
                    keep = np.ones(take, dtype=bool)
                    keep[local] = False
                    m32 = m32[keep]
                m32.tofile(fv)
                kept += m32.shape[0]
                g += take
            log(f"shard {si + 1}/{len(shards)} read: {g} rows so far, {kept} kept")
    if g != used or qfilled != queries or dim is None or qmat is None:
        raise SystemExit(f"read {g} of {used} rows and {qfilled} of {queries} queries: the shards are shorter than their metadata says")
    return tmp, qmat, dim, kept, used, nmin, nmax


def exact_truth(vec_path: Path, count: int, dim: int, q32: np.ndarray, k: int, block_rows: int) -> np.ndarray:
    """Pass 2: the k exact nearest rows of every query. Blocks of vectors give candidates by a float32 matrix product, and the best candidates
    of each query are rescored in float64, so a rounding error of the float32 product cannot reorder the answer."""
    nq = q32.shape[0]
    if count < k:
        raise SystemExit(f"the index holds {count} vectors, fewer than --truth-k {k}")
    pool = min(count, max(2 * k, k + 10))
    block = max(1000, min(block_rows, 20_000_000 // nq))
    best_s = np.full((nq, pool), -np.inf, dtype=np.float32)
    best_i = np.full((nq, pool), -1, dtype=np.int64)
    mm = np.memmap(vec_path, dtype="<f4", mode="r", shape=(count, dim))
    done = 0
    for start in range(0, count, block):
        blk = np.array(mm[start:start + block], dtype=np.float32)
        sc = q32 @ blk.T                                     # (nq, rows of this block), float32
        kk = min(pool, sc.shape[1])
        part = np.argpartition(sc, -kk, axis=1)[:, -kk:]
        cs = np.concatenate([best_s, np.take_along_axis(sc, part, axis=1)], axis=1)
        ci = np.concatenate([best_i, part.astype(np.int64) + start], axis=1)
        sel = np.argpartition(cs, -pool, axis=1)[:, -pool:]
        best_s = np.take_along_axis(cs, sel, axis=1)
        best_i = np.take_along_axis(ci, sel, axis=1)
        done += blk.shape[0]
        if (start // block) % 10 == 0:
            log(f"exact scan: {done}/{count} vectors")
    if (best_i < 0).any():
        raise SystemExit("internal error: a query has fewer candidates than the pool")
    q64 = q32.astype(np.float64)
    result = np.empty((nq, k), dtype="<i4")
    for qi in range(nq):
        ids = np.sort(best_i[qi])
        scores = np.array(mm[ids], dtype=np.float64) @ q64[qi]
        order = np.lexsort((ids, -scores))                   # best score first, the lower row first on a tie
        result[qi] = ids[order][:k]
    del mm
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="a directory of *.parquet files, or one parquet file")
    ap.add_argument("out", type=Path, help="the output directory (created)")
    ap.add_argument("--column", default=DEFAULT_COLUMN, help="the column of embeddings (default: %(default)s)")
    ap.add_argument("--queries", type=int, default=1000, help="queries held out of the set (default: %(default)s)")
    ap.add_argument("--truth-k", type=int, default=10, help="exact answers per query (default: %(default)s)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N rows of the set (a smaller index, with its own exact answers)")
    ap.add_argument("--seed", type=int, default=42, help="seed of the choice of the held-out queries (default: %(default)s)")
    ap.add_argument("--batch-rows", type=int, default=4096, help="rows per read batch (default: %(default)s)")
    ap.add_argument("--block-rows", type=int, default=20000, help="vectors per block of the exact scan (default: %(default)s)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing data set in the output directory")
    args = ap.parse_args()

    out_dir: Path = args.out
    if (out_dir / META).exists() and not args.force:
        raise SystemExit(f"{out_dir / META} exists; pass --force to rebuild it")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / META).unlink(missing_ok=True)

    shards = list_shards(args.src)
    tmp, qmat, dim, count, used, nmin, nmax = read_and_split(shards, args.column, out_dir, args.queries, args.limit, args.seed, args.batch_rows)
    truth = exact_truth(tmp, count, dim, qmat, args.truth_k, args.block_rows)

    qmat.tofile(out_dir / QUERIES)
    truth.tofile(out_dir / TRUTH)
    tmp.replace(out_dir / VECTORS)
    meta = {
        "format": FORMAT,
        "dim": dim,
        "count": count,
        "queries": args.queries,
        "truth-k": args.truth_k,
        "metric": "cosine",
        "normalized": "true",
        "source": args.src.name,
        "column": args.column,
        "source-rows": used,
        "source-norm-min": f"{nmin:.8f}",
        "source-norm-max": f"{nmax:.8f}",
        "heldout": "random",
        "seed": args.seed,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prep-script": "scripts/prep_real_dataset.py",
    }
    with open(out_dir / META, "w", encoding="utf-8") as f:
        for key, value in meta.items():
            f.write(f"{key}={value}\n")
    log(f"done: {count} vectors of dimension {dim}, {args.queries} queries with {args.truth_k} exact answers each, in {out_dir}")


if __name__ == "__main__":
    main()
