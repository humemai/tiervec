#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Prints the tables of docs/EVIDENCE.md from the committed result files. No measured number in those tables is typed.

    uv run scripts/evidence_tables.py             print the tables and the README row
    uv run scripts/evidence_tables.py --write     put the tables between the markers of docs/EVIDENCE.md, and the row into README.md
    uv run scripts/evidence_tables.py --check     exit 1 when the page or the row differs from what the files say

A result file is first checked: it has to end with BENCH-END, report status=OK with no failure and no malformed answer, come from the pinned
commit with no modified file, name no home directory, and have been written with the options that the command on the page names. The files on
real data are also checked for their data set, their cores, and the harness's own comparison of the exact answers. What the text of the page
says about the real data (which differences are under the bar, which cells disagree) is checked too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "EVIDENCE.md"
README = ROOT / "README.md"
BEGIN = "<!-- begin tables: 106 synthetic -->"
END = "<!-- end tables: 106 synthetic -->"
BEGIN_REAL = "<!-- begin tables: 106 real -->"
END_REAL = "<!-- end tables: 106 real -->"
ROW = "| Speed and recall at scale |"

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

DIR_106_REAL = ROOT / "results" / "106" / "real"
PIN_106_REAL = "70b64f2e342430cb0c6389a187c35328219aca2e"
REAL = (
    "--queries 1000 --reps 3 --ef-sweep 20,50,100,200,400,800,1600,3200 --shares 0.01 --scan-below 0 --build-threads 3"
    " --conc-seconds 0"
)
# key: (file stem, short name, heap, the options of the command, data directory, vectors, M, efConstruction)
RUNS_106_REAL = {
    "a0": ("a0-real100k-def", "100k", "13g", REAL, "real100k", "100000", "16", "200"),
    "a1": ("a1-real1m-def", "1M", "13g", REAL, "real1m", "999000", "16", "200"),
    "a2": ("a2-real1m-efc400-m32", "1M: M 32, efConstruction 400", "13g", REAL + " --efc 400 --m 32", "real1m", "999000", "32", "400"),
}
DIM_REAL = "768"
CORES_REAL = "12-17"
# the largest difference of recall@10 at ef 100 that counts as "no drop with size", written before the runs
DROP_BAR = 0.03


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
        elif tag in (
            "RUN", "COMMIT", "CPU", "JDK", "PARAMS", "DATASET", "TRUTH-CHECK", "DATA", "BUILD", "BUILD-STATS", "MEMORY", "CONC", "CONC-CHECK",
            "CONC-RECALL", "BENCH-SUMMARY",
        ):
            run["tags"][tag] = pairs(rest)
    return run


def gate(path: Path, run: dict, heap: str, options: str, pin: str) -> None:
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
    text = path.read_text(encoding="utf-8")
    if f"-Xmx{heap}" not in text:
        problems.append(f"the file does not name -Xmx{heap}")
    if "/home/" in text:
        problems.append("the file names a home directory")
    if problems:
        raise SystemExit(f"{path.stem}: " + "; ".join(problems))


def cells(run: dict, kind: str, ef: str, share: str = "none", scan_first: str = "false") -> list[dict]:
    """The cells of that kind, in the order they ran."""
    return [c for c in run["search"] if c["kind"] == kind and c["ef"] == ef and c["share"] == share and c["scanFirst"] == scan_first]


def cell(run: dict, kind: str, ef: str, share: str = "none", scan_first: str = "false", may_differ: bool = False) -> dict | None:
    """The last cell of that kind. A file whose --ef-sweep names the main --ef has two, and the sweep ran later. Their recall has to agree,
    unless the caller says that the page shows both."""
    found = cells(run, kind, ef, share, scan_first)
    if not may_differ and len({c["recall"] for c in found}) > 1:
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
        gate(DIR_106 / f"{stem}.txt", runs[key], heap, options, PIN_106)

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


def real_runs() -> dict:
    runs = {}
    for key, (stem, _, heap, options, _, n, m, efc) in RUNS_106_REAL.items():
        path = DIR_106_REAL / f"{stem}.txt"
        run = runs[key] = read(path)
        gate(path, run, heap, options, PIN_106_REAL)
        t = run["tags"]
        seen = {
            "vectors": (t["DATA"]["n"], n), "dataset count": (t["DATASET"]["count"], n), "dimensions": (t["DATASET"]["dim"], DIM_REAL),
            "M": (t["PARAMS"]["m"], m), "efConstruction": (t["PARAMS"]["efc"], efc), "cores": (t["CPU"]["cpusAllowedList"], CORES_REAL),
        }
        problems = [f"{what} {got}, expected {want}" for what, (got, want) in seen.items() if got != want]
        if float(t["TRUTH-CHECK"]["fileVsExactScanOverlap"]) < 0.98:
            problems.append("the exact answers of the data files disagree with the harness's scan")
        if problems:
            raise SystemExit(f"{stem}: " + "; ".join(problems))
    if runs["a1"]["tags"]["DATA"]["checksumFirst1000AndQuery0"] != runs["a2"]["tags"]["DATA"]["checksumFirst1000AndQuery0"]:
        raise SystemExit("the two 1M arms ran on different data")
    return runs


def at_100(runs: dict, key: str, which: int) -> dict:
    """One of the two unfiltered cells at ef 100: 0 is the main run, 1 the sweep."""
    found = cells(runs[key], "unfiltered", "100")
    if len(found) != 2:
        raise SystemExit(f"{key}: {len(found)} unfiltered cells at ef 100, the page says two")
    return found[which]


def tables_106_real() -> str:
    runs = real_runs()
    keys = tuple(RUNS_106_REAL)
    names = [RUNS_106_REAL[k][1] for k in keys]

    out = ["### The runs on real data", ""]
    rows = []
    for key, (stem, _, heap, *_) in RUNS_106_REAL.items():
        t = runs[key]["tags"]
        rows.append([
            f"[`{stem}.txt`](../results/106/real/{stem}.txt)", t["RUN"]["started"][:10], t["COMMIT"]["sha"][:10], t["JDK"]["version"],
            t["DATA"]["n"], t["PARAMS"]["queries"], t["PARAMS"]["reps"], t["DATA"]["checksumFirst1000AndQuery0"],
            t["TRUTH-CHECK"]["fileVsExactScanOverlap"], t["CPU"]["cpusAllowedList"], f"`-Xmx{heap}`", t["BENCH-SUMMARY"]["wallSeconds"],
        ])
    out += table(
        ["File", "Date (UTC)", "Commit", "JDK", "Vectors", "Queries", "Timed passes", "Data checksum", "Truth check", "Cores", "Heap",
         "Wall seconds"], rows)

    efs = sorted({c["ef"] for k in keys for c in runs[k]["search"] if c["kind"] == "unfiltered"}, key=int)

    def sweep(key: str, ef: str) -> dict | None:
        return cell(runs[key], "unfiltered", ef, may_differ=ef == "100")

    out += ["### Recall@10 by `ef` on real data", ""]
    out += table(["`ef`"] + names, [[ef] + [recall(sweep(k, ef)) for k in keys] for ef in efs])

    def first_ef(key: str) -> str:
        for ef in efs:
            c = sweep(key, ef)
            if c is not None and float(c["recall"]) >= TARGET:
                return ef
        return "none"

    out += [f"Smallest `ef` of the sweep with recall@10 of at least {TARGET}: " + "; ".join(f"{RUNS_106_REAL[k][1]}: {first_ef(k)}" for k in keys) + ".", ""]
    out += ["### Lowest recall@10 of a single query, by `ef`", ""]
    out += table(["`ef`"] + names, [[ef] + [f"{float(sweep(k, ef)['recallMin']):.2f}" for k in keys] for ef in efs])
    out += ["### p50 ms by `ef` on real data (a direction only)", ""]
    out += table(["`ef`"] + names, [[ef] + [millis(sweep(k, ef)) for k in keys] for ef in efs])

    out += ["### The two cells at `ef` 100", ""]
    rows = []
    for key, name in zip(keys, names):
        main_cell, sweep_cell = at_100(runs, key, 0), at_100(runs, key, 1)
        rows.append([name, recall(main_cell), recall(sweep_cell), millis(main_cell), millis(sweep_cell)])
    out += table(["Arm", "recall@10, main run", "recall@10, sweep", "p50 ms, main run", "p50 ms, sweep"], rows)
    agree = {k: at_100(runs, k, 0)["recall"] == at_100(runs, k, 1)["recall"] for k in keys}
    tails = {k: int(runs[k]["tags"]["BUILD-STATS"]["tail"]) for k in keys}
    if agree != {"a0": False, "a1": True, "a2": True} or not (tails["a0"] > 0 and tails["a1"] == tails["a2"] == 0):
        raise SystemExit(f"the page's text on the two cells or the tail no longer fits the files: agree={agree} tail={tails}")

    def difference(high: str, low: str, which: int) -> float:
        return float(at_100(runs, high, which)["recall"]) - float(at_100(runs, low, which)["recall"])

    out += ["### Differences at `ef` 100", ""]
    out += table(["Difference in recall@10", "sweep cells", "main-run cells"], [
        ["100k minus 1M, default build"] + [f"{difference('a0', 'a1', w):.4f}" for w in (1, 0)],
        ["M 32, efConstruction 400 minus default build, at 1M"] + [f"{difference('a2', 'a1', w):.4f}" for w in (1, 0)],
    ])
    out += [f"The bar for no drop with size, written before the runs: a difference of at most {DROP_BAR} from 100k to 1M.", ""]
    settled = {k: float(runs[k]["tags"]["BUILD"]["settledSeconds"]) for k in keys}
    if not all(difference("a0", "a1", w) <= DROP_BAR and difference("a2", "a1", w) > 0 for w in (0, 1)) or not settled["a2"] > settled["a1"]:
        raise SystemExit("the page's text on the differences or on the settle time no longer fits the files")

    out += ["### Filtered search at `ef` 100, allowed share 0.01", ""]
    filtered = [cell(runs[k], "filtered", "100", "0.01") for k in keys]
    out += table([""] + names, [["recall@10"] + [recall(c) for c in filtered], ["p50 ms (a direction only)"] + [millis(c) for c in filtered]])

    build_rows = (
        ("load seconds of the files", "DATA", "loadSeconds"), ("add seconds", "BUILD", "addSeconds"),
        ("settled seconds", "BUILD", "settledSeconds"), ("seals", "BUILD-STATS", "seals"), ("merges", "BUILD-STATS", "merges"),
        ("segments at the end", "BUILD-STATS", "segments"), ("vectors in the tail at the end of the build", "BUILD-STATS", "tail"),
        ("back-pressure waits", "BUILD-STATS", "backpressureWaits"), ("peak resident MB", "MEMORY", "rssPeakMB"),
        ("wall seconds of the run", "BENCH-SUMMARY", "wallSeconds"),
    )
    out += ["### Build on real data", ""]
    out += table([""] + names, [[label] + [runs[k]["tags"][tag][field] for k in keys] for label, tag, field in build_rows])

    out += ["### Commands on real data", ""]
    for stem, _, heap, options, data, *_ in RUNS_106_REAL.values():
        out += [
            f"`{stem}.txt`:", "", "```",
            f"java -Xmx{heap} --add-modules jdk.incubator.vector -cp target/bench/classes ai.humem.tiervec.Bench \\",
            f"  --data data/{data} --host-label my-machine --out results/{stem}.txt \\",
            f"  {options}",
            "```", "",
        ]
    return "\n".join(out)


def readme_row() -> str:
    """The README's status row on recall at scale. Its numbers come from the result files of both blocks."""
    real = real_runs()
    synthetic = {k: read(DIR_106 / f"{RUNS_106[k][0]}.txt") for k in ("s100k", "s1m")}

    def n(run: dict) -> str:
        return f"{int(run['tags']['PARAMS']['n']):,}"

    def r(run: dict) -> str:
        return recall(cell(run, "unfiltered", "100", may_differ=True))

    a0, a1, s0, s1 = real["a0"], real["a1"], synthetic["s100k"], synthetic["s1m"]
    queries = f"{int(a1['tags']['PARAMS']['queries']):,}"
    text = (
        f"Barely measured: recall at {n(s0)} and about {n(s1)} vectors, one build per arm. Speed is not established, and nothing is measured at"
        f" 10,000,000. On one set of real embeddings (DBpedia with OpenAI embeddings, cut to {a1['tags']['DATASET']['dim']} dimensions,"
        f" {queries} queries) the default build has recall@10 {r(a0)} at {n(a0)} vectors and {r(a1)} at {n(a1)}, at `ef` 100. On generated"
        f" clustered Gaussian data the same defaults have {r(s0)} at {n(s0)} and {r(s1)} at {n(s1)}. The runs used low-power or efficiency"
        " cores and were not timed studies, so their latency columns show a direction only. CI runs the harness at 10,000 vectors as a smoke"
        " test."
    )
    links = "[docs/EVIDENCE.md](docs/EVIDENCE.md), [docs/BENCHMARKS.md](docs/BENCHMARKS.md), [docs/STATUS.md](docs/STATUS.md)"
    return f"{ROW} {text} | {links} |"


def render(page: str) -> str:
    for begin_marker, end_marker, tables in ((BEGIN, END, tables_106), (BEGIN_REAL, END_REAL, tables_106_real)):
        head, begin, rest = page.partition(begin_marker)
        _, end, tail = rest.partition(end_marker)
        if not begin or not end:
            raise SystemExit(f"{PAGE}: the markers {begin_marker} are missing")
        page = f"{head}{begin_marker}\n\n{tables()}\n{end_marker}{tail}"
    return page


def render_readme(readme: str) -> str:
    lines = readme.split("\n")
    at = [i for i, line in enumerate(lines) if line.startswith(ROW)]
    if len(at) != 1:
        raise SystemExit(f"{README}: {len(at)} rows start with {ROW}")
    lines[at[0]] = readme_row()
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not (args.write or args.check):
        print(tables_106(), tables_106_real(), readme_row(), sep="\n")
        return
    stale = []
    for path, renderer in ((PAGE, render), (README, render_readme)):
        old = path.read_text(encoding="utf-8")
        new = renderer(old)
        if args.write:
            path.write_text(new, encoding="utf-8")
        elif new != old:
            stale.append(str(path))
    if stale:
        sys.exit(f"{', '.join(stale)} differs from the result files: run scripts/evidence_tables.py --write")


if __name__ == "__main__":
    main()
