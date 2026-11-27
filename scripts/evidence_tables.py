#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Prints the tables of docs/EVIDENCE.md from the committed result files. No number on that page is typed.

    uv run scripts/evidence_tables.py             print the tables
    uv run scripts/evidence_tables.py --write     put them into docs/EVIDENCE.md, between the two markers
    uv run scripts/evidence_tables.py --check     exit 1 when the page differs from what the files say

A result file is first checked: it has to end with BENCH-END, report status=OK with no failure and no malformed answer, come from the pinned
commit with no modified file, and have been written with the options that the command on the page names.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "EVIDENCE.md"
BEGIN = "<!-- begin tables: 106 synthetic -->"
END = "<!-- end tables: 106 synthetic -->"

DIR_106 = ROOT / "results" / "106" / "synthetic"
PIN_106 = "70d5dee6c373c7cd547ef3c3c9d685e849c3f820"
COMMON_1M = "--n 1000000 --dim 768"
DIAG = COMMON_1M + " --conc-seconds 0 --shares 0.01 --scan-below 0 --ef-sweep 20,50,100,200,400,800,1600,3200"
# key: (file stem, short name, heap, the options of the command)
RUNS_106 = {
    "s100k": (
        "h100k-smoke-jdk25",
        "100k",
        "4g",
        "--n 100000 --dim 768 --queries 100 --reps 3 --shares 0.1,0.01 --scan-below 0,1 --conc-seconds 60 --conc-window-s 15"
        " --write-rate 300 --delete-rate 60 --final-queries 50 --min-recall 0.9 --min-scan-recall 0.99",
    ),
    "s1m": ("h1m-jdk25", "1M", "13g", COMMON_1M + " --conc-seconds 600"),
    "d1": ("h1m-diag-d1-jdk25", "d1: same build again", "13g", DIAG),
    "d2": ("h1m-diag-d2-jdk25", "d2: M 32, efConstruction 400", "13g", DIAG + " --efc 400 --m 32"),
    "d3": ("h1m-diag-d3-jdk25", "d3: one rebuilt segment", "13g", DIAG + " --load rebuild"),
}
TARGET = 0.95


def pairs(text: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in text.split() if "=" in token)


def read(path: Path) -> dict:
    run: dict = {"search": [], "tags": {}, "end": False}
    for line in path.read_text(encoding="utf-8").splitlines():
        tag, _, rest = line.partition(" ")
        if tag == "SEARCH":
            run["search"].append(pairs(rest))
        elif tag == "BENCH-END":
            run["end"] = True
        elif tag in ("RUN", "COMMIT", "JDK", "PARAMS", "DATA", "BUILD", "BUILD-STATS", "CONC", "CONC-CHECK", "CONC-RECALL", "BENCH-SUMMARY"):
            run["tags"][tag] = pairs(rest)
    return run


def gate(stem: str, run: dict, heap: str, options: str, pin: str) -> None:
    problems = []
    tags = run["tags"]
    if not run["end"]:
        problems.append("no BENCH-END")
    summary = tags.get("BENCH-SUMMARY", {})
    if summary.get("status") != "OK" or summary.get("failures") != "0":
        problems.append(f"BENCH-SUMMARY status={summary.get('status')} failures={summary.get('failures')}")
    commit = tags.get("COMMIT", {})
    if commit.get("sha") != pin or commit.get("modifiedFiles") != "0":
        problems.append(f"COMMIT sha={commit.get('sha')} modifiedFiles={commit.get('modifiedFiles')}")
    if any(cell.get("badAnswers") != "0" for cell in run["search"]):
        problems.append("a SEARCH cell has badAnswers")
    params = tags.get("PARAMS", {})
    words = options.split()
    for key, value in zip(words[::2], words[1::2]):
        if params.get(key[2:]) != value:
            problems.append(f"PARAMS {key[2:]}={params.get(key[2:])}, the command on the page says {value}")
    if f"-Xmx{heap}" not in (ROOT / "results" / "106" / "synthetic" / f"{stem}.txt").read_text(encoding="utf-8"):
        problems.append(f"the file does not name -Xmx{heap}")
    if problems:
        raise SystemExit(f"{stem}: " + "; ".join(problems))


def cell(run: dict, kind: str, ef: str, share: str = "none", scan_first: str = "false") -> dict | None:
    """The last cell of that kind. A file whose --ef-sweep names the main --ef has two: the sweep ran later, and their recall has to agree."""
    found = [c for c in run["search"] if c["kind"] == kind and c["ef"] == ef and c["share"] == share and c["scanFirst"] == scan_first]
    if len({c["recall"] for c in found}) > 1:
        raise SystemExit(f"two cells with another recall: {kind} ef={ef} share={share}")
    return found[-1] if found else None


def recall(c: dict | None) -> str:
    return "" if c is None else f"{float(c['recall']):.4f}"


def millis(c: dict | None, key: str = "p50us") -> str:
    return "" if c is None else f"{int(c[key]) / 1000:.2f}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return out + [""]


def tables_106() -> str:
    runs = {}
    for key, (stem, _, heap, options) in RUNS_106.items():
        runs[key] = read(DIR_106 / f"{stem}.txt")
        gate(stem, runs[key], heap, options, PIN_106)

    out = ["### The runs", ""]
    rows = []
    for key, (stem, name, heap, _) in RUNS_106.items():
        t = runs[key]["tags"]
        rows.append([
            f"[`{stem}.txt`](../results/106/synthetic/{stem}.txt)", t["RUN"]["started"][:10], t["COMMIT"]["sha"][:10], t["JDK"]["version"],
            t["PARAMS"]["queries"], t["PARAMS"]["reps"], t["DATA"]["checksumFirst1000AndQuery0"], f"`-Xmx{heap}`", t["BENCH-SUMMARY"]["wallSeconds"],
        ])
    out += table(["File", "Date (UTC)", "Commit", "JDK", "Queries", "Timed passes", "Data checksum", "Heap", "Wall seconds"], rows)

    pair = ("s100k", "s1m")
    names = [RUNS_106[k][1] for k in pair]
    out += ["### Unfiltered search by `ef`: 100,000 against 1,000,000", ""]
    efs = sorted({c["ef"] for k in pair for c in runs[k]["search"] if c["kind"] == "unfiltered"}, key=int)
    rows = []
    for ef in efs:
        cells = [cell(runs[k], "unfiltered", ef) for k in pair]
        rows.append([ef] + [recall(c) for c in cells] + [millis(c) for c in cells])
    out += table(["`ef`"] + [f"recall@10 {n}" for n in names] + [f"p50 ms {n}" for n in names], rows)

    out += ["### Filtered search at `ef` 100, 1,000,000 vectors: walk against scan-first", ""]
    shares = sorted({c["share"] for c in runs["s1m"]["search"] if c["kind"] == "filtered"}, key=float, reverse=True)
    rows = []
    for share in shares:
        walk = cell(runs["s1m"], "filtered", "100", share, "false")
        scan = cell(runs["s1m"], "filtered", "100", share, "true")
        rows.append([share, recall(walk), millis(walk), recall(scan), millis(scan), f"{int(scan['p50us']) / int(walk['p50us']):.2f}"])
    out += table(["Allowed share", "recall walk", "p50 ms walk", "recall scan-first", "p50 ms scan-first", "p50 ratio, scan-first / walk"], rows)

    build_rows = (
        ("add seconds", "BUILD", "addSeconds"), ("settled seconds", "BUILD", "settledSeconds"), ("seals", "BUILD-STATS", "seals"),
        ("merges", "BUILD-STATS", "merges"), ("segments at the end", "BUILD-STATS", "segments"), ("back-pressure waits", "BUILD-STATS", "backpressureWaits"),
    )

    def build_table(keys: tuple[str, ...]) -> list[str]:
        rows = [[label] + [runs[k]["tags"].get(tag, {}).get(field, "") for k in keys] for label, tag, field in build_rows]
        return table([""] + [RUNS_106[k][1] for k in keys], rows)

    out += ["### Build", ""] + build_table(pair)

    conc = runs["s1m"]["tags"]["CONC"]
    check = runs["s1m"]["tags"]["CONC-CHECK"]
    after = runs["s1m"]["tags"]["CONC-RECALL"]
    out += ["### Concurrent phase, 1,000,000 vectors", ""]
    out += table(["", "value"], [
        ["seconds", f"{float(conc['seconds']):.0f}"],
        ["writer threads, search threads", f"{conc['writers']}, {conc['searchers']}"],
        ["adds per second (the writers were throttled to)", f"{conc['addsPerSecond']} ({float(conc['addsTargetPerSecond']):.0f})"],
        ["deletes per second", conc["deletesPerSecond"]],
        ["searches per second", conc["searchesPerSecond"]],
        ["search p50 ms", millis(conc, "searchP50us")],
        ["live vectors at the end, expected", f"{check['size']}, {check['expected']}"],
        [f"recall@10 afterwards, idle, {after['queries']} queries", f"{float(after['recall']):.4f}"],
    ])

    follow = ("s1m", "d1", "d2", "d3")
    names = [RUNS_106[k][1] for k in follow]
    out += ["### Follow-up at 1,000,000 vectors: recall@10 by `ef`", ""]
    efs = sorted({c["ef"] for k in follow for c in runs[k]["search"] if c["kind"] == "unfiltered"}, key=int)
    out += table(["`ef`"] + names, [[ef] + [recall(cell(runs[k], "unfiltered", ef)) for k in follow] for ef in efs])
    out += ["### Follow-up: p50 ms by `ef`", ""]
    out += table(["`ef`"] + names, [[ef] + [millis(cell(runs[k], "unfiltered", ef)) for k in follow] for ef in efs])

    def first_ef(key: str) -> str:
        for ef in efs:
            c = cell(runs[key], "unfiltered", ef)
            if c is not None and float(c["recall"]) >= TARGET:
                return ef
        return "none"

    out += [f"Smallest `ef` of the sweep with recall@10 of at least {TARGET}: " + ", ".join(f"{RUNS_106[k][1]}: {first_ef(k)}" for k in follow) + ".", ""]
    out += ["### Follow-up: build", ""] + build_table(follow)

    out += ["### Commands", ""]
    for _, (stem, _, heap, options) in RUNS_106.items():
        out += [
            f"`{stem}.txt`:", "", "```",
            f"java -Xmx{heap} --add-modules jdk.incubator.vector -cp target/bench/classes ai.humem.tiervec.Bench \\",
            f"  --host-label my-machine --out results/{stem}.txt \\",
            f"  {options}",
            "```", "",
        ]
    return "\n".join(out)


def render(page: str) -> str:
    head, begin, rest = page.partition(BEGIN)
    _, end, tail = rest.partition(END)
    if not begin or not end:
        raise SystemExit(f"{PAGE}: the markers are missing")
    return f"{head}{BEGIN}\n\n{tables_106()}\n{END}{tail}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not (args.write or args.check):
        print(tables_106())
        return
    page = PAGE.read_text(encoding="utf-8")
    new = render(page)
    if args.write:
        PAGE.write_text(new, encoding="utf-8")
    elif new != page:
        sys.exit(f"{PAGE} differs from the result files: run scripts/evidence_tables.py --write")


if __name__ == "__main__":
    main()
