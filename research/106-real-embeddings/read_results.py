#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""The results table of humemai/tiervec#106 for study 106 (README.md), read from the result files, no typed numbers.

    uv run --no-project python -I read_results.py [--results DIR] [--ref-dir DIR] [--allowed 12-17] > analysis.md

It prints, in this order: the gates of every file (a file that fails a gate is a finding, not a measurement, and gets no verdict), recall@10 and p50 by ef per arm (the table of the
issue, with the synthetic runs of 2026-11-12 beside it when --ref-dir has them), the build and settle figures, and the pre-registered reading of README section 5 applied to the
numbers. The thresholds are the ones of the 2026-11-12 follow-up (tvm-diag1m): 0.03 is "repeats / does not move", 0.10 is "moves".
"""
from __future__ import annotations

import argparse
from pathlib import Path

PIN = "70b64f2e342430cb0c6389a187c35328219aca2e"
NOISE, MOVES, TARGET = 0.03, 0.10, 0.95
# arm id: (file stem, label, expected count in the file, expected dim, comparison with the synthetic runs)
ARMS = {
    "a0": ("a0-real100k-def", "real 100k, M16 efc200", 100000, 768),
    "a1": ("a1-real1m-def", "real 1M, M16 efc200", 999000, 768),
    "a2": ("a2-real1m-efc400-m32", "real 1M, M32 efc400", 999000, 768),
    "a3": ("a3-real1m-rebuild", "real 1M, one rebuilt segment", 999000, 768),
    "b1": ("b1-deep1m-def", "DEEP 96-d 1M, M16 efc200", 1000000, 96),
}
REFS = {
    "s100k": ("h100k-smoke-jdk25.txt", "synthetic 100k"),
    "s1m": ("h1m-diag-d1-jdk25.txt", "synthetic 1M"),
    "s1m_big": ("h1m-diag-d2-jdk25.txt", "synthetic 1M M32 efc400"),
}


def kv(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def parse(path: Path) -> dict:
    d: dict = {"path": path, "unf": {}, "bad": 0, "tags": {}}
    for line in path.read_text(errors="replace").splitlines():
        tag = line.split(" ", 1)[0]
        if tag == "SEARCH":
            r = kv(line)
            d["bad"] += int(r.get("badAnswers", "0"))
            if r.get("kind") == "unfiltered":
                d["unf"][int(r["ef"])] = r
        elif tag in ("COMMIT", "BUILD", "BUILD-STATS", "BENCH-SUMMARY", "DATASET", "TRUTH-CHECK", "CPU", "JDK", "MEMORY", "JVM", "DATA"):
            d["tags"].setdefault(tag, []).append(kv(line))
        elif tag == "BENCH-END":
            d["end"] = True
    return d


def expand(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def first(d: dict, tag: str) -> dict:
    return d["tags"].get(tag, [{}])[0]


def gates(arm: str, d: dict, allowed: set[int]) -> list[str]:
    _, _, count, dim = ARMS[arm]
    bad: list[str] = []
    s = first(d, "BENCH-SUMMARY")
    if not d.get("end"):
        bad.append("no BENCH-END (the run did not finish)")
    if s.get("status") != "OK" or s.get("failures") != "0":
        bad.append(f"BENCH-SUMMARY status={s.get('status')} failures={s.get('failures')}")
    c = first(d, "COMMIT")
    if c.get("sha") != PIN or c.get("modifiedFiles") != "0":
        bad.append(f"COMMIT sha={c.get('sha')} modifiedFiles={c.get('modifiedFiles')}, the pin is {PIN} with 0 modified files")
    ds = first(d, "DATASET")
    if ds.get("count") != str(count) or ds.get("dim") != str(dim):
        bad.append(f"DATASET count={ds.get('count')} dim={ds.get('dim')}, expected {count} and {dim}")
    if d["bad"]:
        bad.append(f"badAnswers total {d['bad']}")
    tc = first(d, "TRUTH-CHECK")
    if not tc:
        bad.append("no TRUTH-CHECK line (the harness did not compare the exact answers of the file with its own scan)")
    elif float(tc.get("fileVsExactScanOverlap", "0")) < 0.98:
        bad.append(f"TRUTH-CHECK overlap {tc.get('fileVsExactScanOverlap')} is below 0.98")
    cpu = first(d, "CPU").get("cpusAllowedList", "")
    if not cpu or not expand(cpu) <= allowed:
        bad.append(f"cpusAllowedList={cpu} is not inside the allowed cores")
    if not first(d, "JDK").get("version", "").startswith("25."):
        bad.append(f"JDK {first(d, 'JDK').get('version')} is not 25.x")
    return bad


def recall(d: dict, ef: int) -> float | None:
    r = d["unf"].get(ef)
    return float(r["recall"]) if r else None


def first_ef(d: dict) -> str:
    for ef in sorted(d["unf"]):
        if float(d["unf"][ef]["recall"]) >= TARGET:
            return str(ef)
    return "none"


def fmt(x: float | None, nd: int = 4) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    ap.add_argument("--ref-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "106" / "synthetic")
    ap.add_argument("--allowed", default="12-17", help="cores the run was allowed to use (the benchmark host used 12-17)")
    a = ap.parse_args()
    allowed = expand(a.allowed)

    arms = {k: parse(a.results / f"{v[0]}.txt") for k, v in ARMS.items() if (a.results / f"{v[0]}.txt").exists()}
    refs = {k: parse(a.ref_dir / v[0]) for k, v in REFS.items() if (a.ref_dir / v[0]).exists()}
    if not arms:
        raise SystemExit(f"no result file of study 106 in {a.results}")

    print("# Study 106: recall@10 of the tier on real embeddings (generated by read_results.py)\n")
    print("## Gates\n")
    print("| arm | file | gates |\n|---|---|---|")
    failed = False
    for k, d in arms.items():
        bad = gates(k, d, allowed)
        failed |= bool(bad)
        print(f"| {k} {ARMS[k][1]} | {d['path'].name} | {'all passed' if not bad else 'FAILED: ' + '; '.join(bad)} |")
    print()

    cols = [(k, ARMS[k][1], d) for k, d in arms.items()] + [(k, REFS[k][1], d) for k, d in refs.items()]
    efs = sorted({ef for _, _, d in cols for ef in d["unf"]})
    print("## Unfiltered search by ef\n")
    print("| ef | " + " | ".join(f"recall {n}" for _, n, _ in cols) + " | " + " | ".join(f"p50 ms {n}" for _, n, _ in cols) + " |")
    print("|---:|" + "---:|" * (2 * len(cols)))
    for ef in efs:
        rec = [fmt(recall(d, ef)) for _, _, d in cols]
        p50 = [fmt(int(d["unf"][ef]["p50us"]) / 1000, 2) if ef in d["unf"] else "" for _, _, d in cols]
        print(f"| {ef} | " + " | ".join(rec) + " | " + " | ".join(p50) + " |")
    print("\nSmallest ef with recall at least 0.95: " + ", ".join(f"{n}: {first_ef(d)}" for _, n, d in cols) + ".\n")

    print("## Build and settle\n")
    print("| | " + " | ".join(n for _, n, _ in cols) + " |\n|---|" + "---:|" * len(cols))
    for label, tag, field in (("load seconds of the files", "DATA", "loadSeconds"), ("add seconds", "BUILD", "addSeconds"), ("settled seconds", "BUILD", "settledSeconds"),
                              ("seals", "BUILD-STATS", "seals"), ("merges", "BUILD-STATS", "merges"), ("segments at the end", "BUILD-STATS", "segments"),
                              ("back-pressure waits", "BUILD-STATS", "backpressureWaits"), ("peak resident MB", "MEMORY", "rssPeakMB"), ("wall seconds", "BENCH-SUMMARY", "wallSeconds")):
        cells = []
        for _, _, d in cols:
            rows = d["tags"].get(tag, [])
            vals = [r[field] for r in rows if field in r]
            cells.append(vals[-1] if vals else "")
        print(f"| {label} | " + " | ".join(cells) + " |")
    print()

    print("## Reading (README section 5, rules written before any run)\n")
    if failed:
        print("A gate failed: the files above are findings to explain, not measurements. No verdict is read from them.")
        return
    r0, r1, r2 = (recall(arms[k], 100) if k in arms else None for k in ("a0", "a1", "a2"))
    if r0 is None or r1 is None:
        print("Needs a0 and a1 at ef=100: no verdict yet.")
        return
    drop = r0 - r1
    print(f"- D = recall(a0, ef=100) - recall(a1, ef=100) = {r0:.4f} - {r1:.4f} = {drop:.4f}")
    if r2 is not None:
        gain = r2 - r1
        print(f"- P = recall(a2, ef=100) - recall(a1, ef=100) = {r2:.4f} - {r1:.4f} = {gain:.4f}")
    else:
        gain = None
        print("- P: a2 has not run.")
    if all(k in refs for k in ("s100k", "s1m", "s1m_big")):
        sd = recall(refs["s100k"], 100) - recall(refs["s1m"], 100)
        sp = recall(refs["s1m_big"], 100) - recall(refs["s1m"], 100)
        print(f"- the synthetic runs of 2026-11-12 (files in --ref-dir): D = {sd:.4f}, P = {sp:.4f}")
    if drop <= NOISE:
        verdict = "DATA: no drop with size on real embeddings (D at most 0.03). The default build holds at 1M here; the 0.58 of the synthetic data is a property of that distribution."
    elif gain is None:
        verdict = "OPEN: D is above 0.03 and a2 has not run."
    elif drop > MOVES and gain > MOVES:
        verdict = "PARAMETERS: the drop exists on real embeddings (D above 0.10) and a stronger graph removes much of it (P above 0.10). The data sets the size of the effect, not its existence."
    elif drop > MOVES and gain <= NOISE:
        verdict = "OTHER: the drop exists on real embeddings and M and efConstruction do not move it (P at most 0.03). Neither the data nor these two parameters: run a3 and read the segment policy (issue #4)."
    else:
        verdict = "UNCLEAR: D or P is between 0.03 and 0.10. The one declared confirmatory rerun of a1 is allowed (README section 4); if it stays in the band the answer is 'small effect'."
    print(f"- Verdict by the rule: {verdict}")
    if "a3" in arms and recall(arms["a3"], 100) is not None:
        print(f"- a3 (one rebuilt segment) against a1 at ef=100: {recall(arms['a3'], 100) - r1:+.4f} (more than 0.10 would mean the merge policy matters, at most 0.03 that it does not)")
    print("\nLatency columns are directional: untimed bulk lane, other jobs may have run (the .env and .samples files beside each result name what did).")


if __name__ == "__main__":
    main()
