#!/usr/bin/env python3
"""aa_check.py RUNDIR: validity (A/A) and per-cell outcome of one timed run of jv-int8-r934c13, from RUNDIR/ab.txt (written by to_jsonl_java.py --clean). Nothing is typed: every number is read.
The rule was written on 2026-11-05, before any run; PLAN.md states it for this study.

Arms: A = jv-FP32, B = jv-INT8, Z = jv-FP32-aa (A again, same build). Figures read: lat50 of B vs A (gating) and lat50 of Z vs A (A/A control).
INVALID (exit 2, report nothing, rerun with a new OUT): any 'COLLISION:' in a lat50 block of B vs A or Z vs A; the median of Z vs A lat50 outside [0.97, 1.03] with >= 10 clean pairs; a figure block missing.
WEAK (exit 3, report nothing, rerun): fewer than 10 clean pairs on either lat50 figure.
Otherwise exit 0 and the outcome of the cell, by the p50 interval of B vs A (rule item 2 of the first run): INT8-FASTER (upper end <= 1.00), INT8-SLOWER (lower end > 1.00), GRAY (the interval contains 1.00 and its upper end is above 1.00).
The outcome is only the p50 half of the README rule: the memory credit and the recall conditions are read by make_verdict. A GRAY cell of a deciding cell (sift128, 1536-d) gets the one declared confirmatory rerun (OUT ...-r2).
Writes RUNDIR/validity.txt (one line: 'ok <outcome> ...' or 'INVALID ...' or 'WEAK ...') and prints it. NIGHT_DRY=1 (the dry run, one round): prints 'DRY' and exits 0 without judging."""
import os
import re
import sys

BAND = (0.97, 1.03)
MINPAIRS = 10
GATE = "lat50-jv-FP32-vs-jv-INT8"
AA = "lat50-jv-FP32-vs-jv-FP32-aa"


def blocks(text):
    out = {}
    for blk in text.split("== ")[1:]:
        name = blk.splitlines()[0].strip()
        out[name] = blk
    return out


def figure(blk):
    n = re.search(r"clean pairs (\d+)", blk)
    m = re.search(r"median ratio B vs A ([\d.]+)\s+95% CI \[([\d.]+), ([\d.]+)\]", blk)
    return {"n": int(n.group(1)) if n else 0, "med": float(m.group(1)) if m else None, "lo": float(m.group(2)) if m else None,
            "hi": float(m.group(3)) if m else None, "collision": "COLLISION:" in blk}


def judge(text):
    """-> (exit code, one line)."""
    b = blocks(text)
    missing = [k for k in (GATE, AA) if k not in b]
    if missing:
        return 2, "INVALID figure block missing: " + ", ".join(missing)
    g, z = figure(b[GATE]), figure(b[AA])
    why = []
    if g["collision"] or z["collision"]:
        why.append("COLLISION in " + ", ".join(k for k, x in ((GATE, g), (AA, z)) if x["collision"]))
    if z["med"] is not None and z["n"] >= MINPAIRS and not (BAND[0] <= z["med"] <= BAND[1]):
        why.append(f"A/A median {z['med']:.3f} of lat50 outside [{BAND[0]}, {BAND[1]}] ({z['n']} clean pairs)")
    if why:
        return 2, "INVALID " + "; ".join(why)
    weak = [f"{k}: {x['n']} clean pairs (< {MINPAIRS})" for k, x in ((GATE, g), (AA, z)) if x["n"] < MINPAIRS or x["med"] is None]
    if weak:
        return 3, "WEAK " + "; ".join(weak)
    if g["hi"] <= 1.0:
        oc = "INT8-FASTER"
    elif g["lo"] > 1.0:
        oc = "INT8-SLOWER"
    else:
        oc = "GRAY"
    return 0, (f"ok {oc} p50 B/A {g['med']:.3f} [{g['lo']:.3f}, {g['hi']:.3f}] ({g['n']} clean pairs); "
               f"A/A Z/A {z['med']:.3f} [{z['lo']:.3f}, {z['hi']:.3f}] ({z['n']} clean pairs) in [{BAND[0]}, {BAND[1]}]")


def main():
    run = os.path.abspath(sys.argv[1])
    if os.environ.get("NIGHT_DRY"):
        print("DRY: validity not judged")
        return 0
    path = os.path.join(run, "ab.txt")
    if not os.path.exists(path):
        line, rc = "INVALID ab.txt missing", 2
    else:
        rc, line = judge(open(path, errors="replace").read())
    open(os.path.join(run, "validity.txt"), "w").write(line + "\n")
    print(line)
    return rc


if __name__ == "__main__":
    sys.exit(main())
