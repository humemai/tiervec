#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "pyarrow>=15"]
# ///
"""Data preparation of study 106 (README.md): the three data sets of the study, in the format of scripts/prep_real_dataset.py (humemai/tiervec, the loader RealData.java).

    uv run --python 3.12 prep_set.py dbpedia SHARDS_DIR OUT --prep-script PATH   # 1,000,000 rows of DBpedia OpenAI text-embedding-3-large, first 768 coordinates, renormalised
    uv run --python 3.12 prep_set.py subset  SRC OUT --n 100000 --prep-script PATH   # a seeded random subset of a prepared set, same queries, new exact answers
    uv run --python 3.12 prep_set.py deep    DEEP_BASE_NPY OUT --prep-script PATH    # control: the first 1,000,000 rows of DEEP (96 dimensions), 1,000 next rows as queries
    uv run --python 3.12 prep_set.py check   DIR                                     # an independent exact check of the files of a prepared set

``--prep-script`` is scripts/prep_real_dataset.py of the pinned commit (`git show SHA:scripts/prep_real_dataset.py`). It is imported, not copied, and its sha256 must be the pinned
one, so the streaming, the normalisation and the exact scan are the code of the repository. Only two things differ: ``dbpedia`` keeps the first ``--dim`` coordinates of every
vector before the normalisation (the shortening that text-embedding-3-large is trained for, the same rule as the 768-dimension arm of datastax/jvector#709), and ``subset`` and
``deep`` build their files from a file that is already on disk. ``meta.properties`` is written last in every case, so a directory without it is a run that did not finish.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

T0 = time.time()
PIN_PREP_SHA256 = "4c6b1cf2041d5d55f6ae501666dec53b3fd61e016f76202398c2ff9054c821e3"  # scripts/prep_real_dataset.py at 70b64f2e342430cb0c6389a187c35328219aca2e
VECTORS, QUERIES, TRUTH, META = "vectors.f32", "queries.f32", "truth.i32", "meta.properties"
SOURCE_DIM = 1536
EXPECT_ROWS = 1_000_000


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def die(msg: str) -> "None":
    raise SystemExit(f"prep_set.py: {msg}")


def load_prep(path: str):
    p = Path(path)
    if not p.is_file():
        die(f"--prep-script {p} does not exist (git show SHA:scripts/prep_real_dataset.py > FILE)")
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != PIN_PREP_SHA256:
        die(f"{p} has sha256 {got}, the pin is {PIN_PREP_SHA256}")
    spec = importlib.util.spec_from_file_location("prep_real_dataset", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prep_real_dataset"] = mod
    spec.loader.exec_module(mod)
    return mod


def read_meta(d: Path) -> dict[str, str]:
    mp = d / META
    if not mp.is_file():
        die(f"{d} has no {META}")
    meta: dict[str, str] = {}
    for line in mp.read_text(encoding="utf-8").splitlines():
        if line.strip() and "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()
    return meta


def write_meta(d: Path, meta: dict) -> None:
    tmp = d / (META + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for key, value in meta.items():
            if "\n" in str(value) or "\\" in str(value):
                die(f"meta value of {key} has a newline or a backslash")
            f.write(f"{key}={value}\n")
    tmp.replace(d / META)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def start_out(out: Path, force: bool) -> None:
    if (out / META).exists() and not force:
        die(f"{out / META} exists; pass --force to rebuild it")
    out.mkdir(parents=True, exist_ok=True)
    (out / META).unlink(missing_ok=True)


def cmd_dbpedia(a: argparse.Namespace) -> None:
    prep = load_prep(a.prep_script)
    out = Path(a.out)
    start_out(out, a.force)
    keep = a.dim
    original = prep.matrix_of

    def truncated(column, name):
        m = original(column, name)
        if m.shape[1] != SOURCE_DIM:
            die(f"the embeddings have {m.shape[1]} dimensions, the rule is written for {SOURCE_DIM}")
        return np.ascontiguousarray(m[:, :keep])

    prep.matrix_of = truncated  # read_and_split looks the name up in the module at call time
    shards = prep.list_shards(Path(a.src))
    tmp, qmat, dim, count, used, nmin, nmax = prep.read_and_split(shards, a.column, out, a.queries, None, a.seed, a.batch_rows)
    if used != a.expect_rows or dim != keep:
        die(f"read {used} rows of dimension {dim}, expected {a.expect_rows} of {keep}")
    truth = prep.exact_truth(tmp, count, dim, qmat, a.truth_k, a.block_rows)
    qmat.tofile(out / QUERIES)
    truth.tofile(out / TRUTH)
    tmp.replace(out / VECTORS)
    write_meta(out, {
        "format": 1, "dim": dim, "count": count, "queries": a.queries, "truth-k": a.truth_k, "metric": "cosine", "normalized": "true",
        "source": Path(a.src).name, "column": a.column, "source-rows": used,
        "source-norm-min": f"{nmin:.8f}", "source-norm-max": f"{nmax:.8f}",
        "truncate-source-dim": SOURCE_DIM, "truncate-dim": keep,
        "truncate-rule": f"the first {keep} coordinates of every vector, then scaled to length 1 (norms before scaling are those of the first {keep} coordinates)",
        "heldout": "random", "seed": a.seed, "created": now(),
        "prep-script": f"study-106 prep_set.py dbpedia over scripts/prep_real_dataset.py sha256 {PIN_PREP_SHA256}",
    })
    log(f"done: {count} vectors of dimension {dim}, {a.queries} queries with {a.truth_k} exact answers each, in {out}")


def cmd_subset(a: argparse.Namespace) -> None:
    prep = load_prep(a.prep_script)
    src, out = Path(a.src), Path(a.out)
    meta = read_meta(src)
    dim, count, nq, k = int(meta["dim"]), int(meta["count"]), int(meta["queries"]), int(meta["truth-k"])
    if not 0 < a.n < count:
        die(f"--n {a.n} must be above 0 and below the {count} vectors of {src}")
    start_out(out, a.force)
    ids = np.sort(np.random.default_rng(a.seed).choice(count, size=a.n, replace=False))
    mm = np.memmap(src / VECTORS, dtype="<f4", mode="r", shape=(count, dim))
    tmp = out / (VECTORS + ".tmp")
    with open(tmp, "wb") as f:
        for s in range(0, a.n, 50_000):
            np.ascontiguousarray(mm[ids[s:s + 50_000]]).tofile(f)
    del mm
    q = np.fromfile(src / QUERIES, dtype="<f4")
    if q.size != nq * dim:
        die(f"{src / QUERIES} holds {q.size} floats, the header says {nq * dim}")
    q = q.reshape(nq, dim)
    log(f"subset of {a.n} of {count} vectors written, scanning for the exact answers of {nq} queries")
    truth = prep.exact_truth(tmp, a.n, dim, q, k, a.block_rows)
    shutil.copyfile(src / QUERIES, out / QUERIES)
    truth.tofile(out / TRUTH)
    tmp.replace(out / VECTORS)
    new = dict(meta)
    new.update({
        "count": a.n, "created": now(), "subset-of": src.name, "subset-source-count": count, "subset-seed": a.seed,
        "subset-rule": "a uniform random subset of the rows of the source file (numpy default_rng(seed).choice without replacement, rows kept in file order); the queries are the source queries, none is in the source index",
        "subset-source-meta-sha256": hashlib.sha256((src / META).read_bytes()).hexdigest(),
        "prep-script": f"study-106 prep_set.py subset over scripts/prep_real_dataset.py sha256 {PIN_PREP_SHA256}",
    })
    write_meta(out, new)
    log(f"done: {a.n} vectors of dimension {dim}, the same {nq} queries, exact answers rescanned, in {out}")


def cmd_deep(a: argparse.Namespace) -> None:
    prep = load_prep(a.prep_script)
    base = np.load(a.src, mmap_mode="r")
    if base.dtype != np.float32 or base.ndim != 2:
        die(f"{a.src} is {base.dtype} with {base.ndim} dimensions, expected a float32 matrix")
    rows, dim = base.shape
    if rows < a.n + a.queries:
        die(f"{a.src} has {rows} rows, fewer than {a.n} + {a.queries}")
    out = Path(a.out)
    start_out(out, a.force)
    tmp = out / (VECTORS + ".tmp")
    nmin, nmax = float("inf"), 0.0
    with open(tmp, "wb") as f:
        for s in range(0, a.n, 100_000):
            m32, lo, hi = prep.normalise(np.asarray(base[s:min(s + 100_000, a.n)], dtype=np.float64))
            nmin, nmax = min(nmin, lo), max(nmax, hi)
            m32.tofile(f)
    qmat, _, _ = prep.normalise(np.asarray(base[a.n:a.n + a.queries], dtype=np.float64))
    truth = prep.exact_truth(tmp, a.n, dim, qmat, a.truth_k, a.block_rows)
    qmat.tofile(out / QUERIES)
    truth.tofile(out / TRUTH)
    tmp.replace(out / VECTORS)
    write_meta(out, {
        "format": 1, "dim": dim, "count": a.n, "queries": a.queries, "truth-k": a.truth_k, "metric": "cosine", "normalized": "true",
        "source": Path(a.src).name, "source-rows": rows, "source-norm-min": f"{nmin:.8f}", "source-norm-max": f"{nmax:.8f}",
        "heldout": "next-rows",
        "cut-rule": f"index is rows 0 to {a.n - 1} of the file, the queries are rows {a.n} to {a.n + a.queries - 1} (held out, not in the index), the dataset's own queries and neighbours are not used",
        "created": now(), "prep-script": f"study-106 prep_set.py deep over scripts/prep_real_dataset.py sha256 {PIN_PREP_SHA256}",
    })
    log(f"done: {a.n} vectors of dimension {dim}, {a.queries} queries with {a.truth_k} exact answers each, in {out}")


def cmd_check(a: argparse.Namespace) -> None:
    d = Path(a.dir)
    meta = read_meta(d)
    dim, count, nq, k = int(meta["dim"]), int(meta["count"]), int(meta["queries"]), int(meta["truth-k"])
    for name, want in ((VECTORS, count * dim * 4), (QUERIES, nq * dim * 4), (TRUTH, nq * k * 4)):
        have = (d / name).stat().st_size
        if have != want:
            die(f"{d / name} is {have} bytes, the header says {want}")
    mm = np.memmap(d / VECTORS, dtype="<f4", mode="r", shape=(count, dim))
    rng = np.random.default_rng(a.seed)
    rows = np.sort(rng.choice(count, size=min(a.norm_sample, count), replace=False))
    norms = np.linalg.norm(np.asarray(mm[rows], dtype=np.float64), axis=1)
    worst = float(np.abs(norms - 1.0).max())
    truth = np.fromfile(d / TRUTH, dtype="<i4").reshape(nq, k)
    if truth.min() < 0 or truth.max() >= count:
        die(f"{d / TRUTH} has an id outside 0 to {count - 1}")
    if any(len(set(row.tolist())) != k for row in truth):
        die(f"{d / TRUTH} has a query with a repeated id")
    qi = np.sort(rng.choice(nq, size=min(a.sample, nq), replace=False))
    q = np.fromfile(d / QUERIES, dtype="<f4").reshape(nq, dim)[qi].astype(np.float64)
    best_s = np.full((len(qi), k), -np.inf)
    best_i = np.full((len(qi), k), -1, dtype=np.int64)
    for s in range(0, count, 20_000):
        sc = q @ np.asarray(mm[s:s + 20_000], dtype=np.float64).T
        kk = min(k, sc.shape[1])
        part = np.argpartition(sc, -kk, axis=1)[:, -kk:]
        cs = np.concatenate([best_s, np.take_along_axis(sc, part, axis=1)], axis=1)
        ci = np.concatenate([best_i, part.astype(np.int64) + s], axis=1)
        sel = np.argsort(-cs, axis=1, kind="stable")[:, :k]
        best_s, best_i = np.take_along_axis(cs, sel, axis=1), np.take_along_axis(ci, sel, axis=1)
    overlaps = [len(set(best_i[j].tolist()) & set(truth[qi[j]].tolist())) / k for j in range(len(qi))]
    mean, low = float(np.mean(overlaps)), float(np.min(overlaps))
    dup = int((best_s[:, 0] > 1.0 - 1e-9).sum())
    log(f"CHECK {d.name}: count={count} dim={dim} queries={nq} sampled_queries={len(qi)} exact_overlap_mean={mean:.4f} exact_overlap_min={low:.4f} "
        f"worst_norm_error_of_{len(rows)}_rows={worst:.2e} queries_equal_to_an_index_row={dup}")
    if mean < 0.99 or worst > 1e-3:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser, prep: bool = True) -> None:
        if prep:
            p.add_argument("--prep-script", required=True, help="scripts/prep_real_dataset.py of the pinned commit")
        p.add_argument("--truth-k", type=int, default=10)
        p.add_argument("--block-rows", type=int, default=20000)
        p.add_argument("--force", action="store_true")

    p = sub.add_parser("dbpedia")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--column", default="text-embedding-3-large-1536-embedding")
    p.add_argument("--expect-rows", type=int, default=EXPECT_ROWS, help="rows the shards must hold (default %(default)s: the whole set; a smoke test passes less)")
    p.add_argument("--batch-rows", type=int, default=4096)
    common(p)
    p.set_defaults(fn=cmd_dbpedia)

    p = sub.add_parser("subset")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    common(p)
    p.set_defaults(fn=cmd_subset)

    p = sub.add_parser("deep")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--queries", type=int, default=1000)
    common(p)
    p.set_defaults(fn=cmd_deep)

    p = sub.add_parser("check")
    p.add_argument("dir", type=Path)
    p.add_argument("--sample", type=int, default=20, help="queries whose exact answers are recomputed in float64 (default %(default)s)")
    p.add_argument("--norm-sample", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_check)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
