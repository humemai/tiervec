#!/usr/bin/env python3
"""report.py STUDYDIR OUT [OUT ...] [--earlier DIR]: markdown tables for the post, generated from the result files only (nothing is typed).

STUDYDIR/OUT/<cell>/ab/{ab.txt,validity.txt,arms.txt,provider.txt} and STUDYDIR/OUT/<cell>/prep/results.jsonl, STUDYDIR/OUT/env.txt.
Figure names in ab.txt: <stat>-jv-FP32-vs-jv-INT8 (B = INT8 against A = FP32) and <stat>-jv-FP32-vs-jv-FP32-aa (the A/A control). The ratio is B over A, below 1.00 means B is faster
(for the throughput figures ab.py already uses A over B, so the direction is the same). --earlier DIR reads DIR/<cell>/ab/ab.txt of the posted run (layer1-out-int8) for the lines marked 'posted'.
Reporting file: not in SHA256SUMS (README point 8); every number it prints is a number of a result file."""
import json
import os
import re
import sys

STATS = [("lat50", "latency p50"), ("lat95", "latency p95"), ("lat99", "latency p99"), ("qps1", "throughput, 1 thread"), ("qpsN", "throughput, 2 threads")]
DIMS = {"sift128": 128, "dbpedia-openai3-1536": 1536, "dbpedia-openai3-768": 768, "dbpedia-openai3-256": 256}


def parse_ab(path):
    out = {}
    if not os.path.exists(path):
        return out
    for blk in open(path).read().split("== ")[1:]:
        name = blk.splitlines()[0].strip()
        n = re.search(r"clean pairs (\d+)", blk)
        m = re.search(r"median ratio B vs A ([\d.]+)\s+95% CI \[([\d.]+), ([\d.]+)\]", blk)
        out[name] = {"n": int(n.group(1)) if n else 0, "med": float(m.group(1)) if m else None, "lo": float(m.group(2)) if m else None, "hi": float(m.group(3)) if m else None}
    return out


def fmt(f):
    return "n/a" if not f or f["med"] is None else f"{f['med']:.3f} [{f['lo']:.3f}, {f['hi']:.3f}]"


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def cell_report(study, out, cell, earlier):
    base = os.path.join(study, out, cell)
    ab = parse_ab(os.path.join(base, "ab", "ab.txt"))
    res = rows(os.path.join(base, "prep", "results.jsonl"))
    val = open(os.path.join(base, "ab", "validity.txt")).read().strip() if os.path.exists(os.path.join(base, "ab", "validity.txt")) else "no validity.txt"
    prov = open(os.path.join(base, "ab", "provider.txt")).read().strip() if os.path.exists(os.path.join(base, "ab", "provider.txt")) else ""
    posted = parse_ab(os.path.join(earlier, cell, "ab", "ab.txt")) if earlier else {}
    L = [f"### {cell} ({out})", "", f"validity: {val}", f"{prov}", ""]
    hdr = "| figure | INT8 / FP32 | clean pairs | A/A (FP32 again / FP32) | clean pairs |" + (" posted run (575dcee1b, laptop) | clean pairs |" if posted else "")
    L += [hdr, "|---|---|---|---|---|" + ("---|---|" if posted else "")]
    for st, label in STATS:
        b, z, p = ab.get(f"{st}-jv-FP32-vs-jv-INT8"), ab.get(f"{st}-jv-FP32-vs-jv-FP32-aa"), posted.get(f"{st}-jv-FP32-vs-jv-INT8")
        L.append(f"| {label} | {fmt(b)} | {b['n'] if b else 'n/a'} | {fmt(z)} | {z['n'] if z else 'n/a'} |" + (f" {fmt(p)} | {p['n'] if p else 'n/a'} |" if posted else ""))
    L.append("")
    by = {}
    for r in res:
        by.setdefault(r["arm"], {}).setdefault(r["phase"], []).append(r)
    pres = rows(os.path.join(earlier, cell, "results.jsonl")) if earlier else []
    pc = {r["arm"]: r for r in pres if r.get("phase") == "calib"}
    L += ["| arm | efSearch | rerank | recall@10 | recall@10 at efSearch 50, rerank 0 | stored vectors (MB) | index file (MB) | peak RSS of the query process (MB) | build (s, information) |" + (" posted run: efSearch, rerank | posted run: recall@10 |" if pres else ""), "|---|---|---|---|---|---|---|---|---|" + ("---|---|" if pres else "")]
    for arm in ("jv-FP32", "jv-INT8"):
        a = by.get(arm, {})
        c = a.get("calib", [{}])[-1]
        q50 = [r for r in a.get("query", []) if r.get("ef") == 50 and r.get("rerank") == 0]
        bld = a.get("build", [{}])[-1]
        mem = a.get("mem", [{}])[-1]
        pk = a.get("peak", [{}])[-1]
        stored = (bld.get("int8_bytes") if arm == "jv-INT8" else bld.get("fp32_bytes"))
        q = pc.get(arm, {})
        tail = f" {q.get('ef', 'n/a')}, {q.get('rerank', 'n/a')} | {q.get('recall10', 'n/a')} |" if pres else ""
        L.append(f"| {arm} | {c.get('ef', 'n/a')} | {c.get('rerank', 'n/a')} | {c.get('recall10', 'n/a')} | {q50[0]['recall10'] if q50 else 'n/a'} | {stored / 1e6:.1f} | {mem.get('index_bytes', 0) / 1e6:.1f} | {pk.get('peak_rss_mb', 'n/a')} | {bld.get('build_s', 'n/a')} |" + tail
                 if stored else f"| {arm} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |" + tail)
    infos = sorted({(r["arm"], r.get("vectorization")) for r in res if r.get("phase") == "info"})
    L += ["", "vectorization provider in prep: " + ", ".join(f"{a}: {v}" for a, v in infos), ""]
    return "\n".join(L)


def main():
    a = sys.argv[1:]
    earlier = a[a.index("--earlier") + 1] if "--earlier" in a else ""
    if "--earlier" in a:
        i = a.index("--earlier")
        a = a[:i] + a[i + 2:]
    study, outs = a[0], a[1:]
    for out in outs:
        envp = os.path.join(study, out, "env.txt")
        print(f"## {out}\n")
        if os.path.exists(envp):
            keep = ("started", "host", "kernel", "Cpus_allowed_list", "HUMEM_HOST_LANE", "HUMEM_HOST_CPUS", "model name", "governor", "no_turbo", "provider", "jdk", "jvector jar")
            print("\n".join("    " + l for l in open(envp).read().splitlines() if l.split(" ")[0].rstrip(":") in keep or l.startswith(("model name", "jvector jar", "jdk"))), "\n")
        for cell in sorted(d for d in os.listdir(os.path.join(study, out)) if os.path.isdir(os.path.join(study, out, d))):
            print(cell_report(study, out, cell, earlier))


if __name__ == "__main__":
    main()
