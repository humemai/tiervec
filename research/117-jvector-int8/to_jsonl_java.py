#!/usr/bin/env python3
# REMOVE WHEN: the layer-1 timed stage is decided (see run_ab_java.sh).
"""to_jsonl_java.py DIR [--clean]: DIR/<arm>.<metric>.<round>.json (JMH) + .meta -> DIR/fig-<stat>-<A>-vs-<B>.jsonl for tools/ab.py (A = baseline).
Metrics: lat -> lat50 / lat95 / lat99 (us, lower is better, primaryMetric.scorePercentiles of the sample-mode run), qps1 / qpsN (ops/s, higher is better), build / ingest (s, lower is better).
Extra figures read from each run's log: rss / rssanon / rssfile (steady RSS after the warm pass of the lat run, MB), buildpeakrss / buildpeakrssanon (peak RSS during the build, MB), buildcpu (process CPU seconds of the build), buildtotal (build + ingest).
Pairs (--dim D adds the bucket pairs; --main adds jv-BQ vs ad-BINARY): jv-FP32 vs each jv arm, ad-NONE vs each ad arm, jv-FP32 vs ad-NONE, jv-PQ<S> vs ad-PQ<S>. --clean also runs ab.py --noise per figure and prints 'MINCLEAN n', then each figure's verdict lines."""
import glob, json, os, re, subprocess, sys
d = sys.argv[1]
dim = int(sys.argv[sys.argv.index("--dim") + 1]) if "--dim" in sys.argv else 0
EXTRA = "--extra" in sys.argv   # the figures read from the logs (rss, peak RSS, CPU, buildtotal): only the lucene and build studies; the existing studies' figures and clean-pair counts stay exactly as before
vals = {}; extra = {}   # extra: figures read from the run's log (RESULT lines): steady RSS of the lat run, peak RSS and process CPU of the build run


def result_rows(path):
    out = []
    try:
        for line in open(path, errors="replace"):
            m = re.search(r"RESULT (\{.*\})\s*$", line)
            if m:
                try: out.append(json.loads(m.group(1)))
                except ValueError: pass
    except OSError: pass
    return out
   # stat -> {(arm, round): (value, meta)}
for f in glob.glob(f"{d}/*.*.*.json"):
    arm, metric, k = os.path.basename(f).split(".")[:3]
    if not os.path.exists(f[:-5] + ".meta"): continue
    try: res = json.load(open(f))
    except ValueError: continue
    meta = json.load(open(f[:-5] + ".meta")); m = res[0]["primaryMetric"]
    stats = ({"lat50": m["scorePercentiles"]["50.0"], "lat95": m["scorePercentiles"]["95.0"], "lat99": m["scorePercentiles"]["99.0"]}
             if metric == "lat" else {metric: m["score"]})
    for s, v in stats.items(): vals.setdefault(s, {})[(arm, int(k))] = (v, meta)
    for r in (result_rows(f[:-5] + ".log") if EXTRA else []):
        if metric == "lat" and r.get("phase") == "bench-mem":   # steady-state RSS after the warm query pass (anon = heap + native, file = mapped index files)
            for key, st in (("rss_mb", "rss"), ("rss_anon_mb", "rssanon"), ("rss_file_mb", "rssfile")):
                if key in r: vals.setdefault(st, {})[(arm, int(k))] = (r[key], meta)
        if metric in ("build", "ingest") and str(r.get("phase", "")).startswith("bench-build-"):
            for key, st in (("peak_rss_mb", "peakrss"), ("peak_rss_anon_mb", "peakrssanon"), ("build_cpu_s", "cpu")):
                if key in r: vals.setdefault(f"{metric}-{st}" if metric == "ingest" else f"build{st}", {})[(arm, int(k))] = (r[key], meta)
# total build = ingest + index for ArcadeDB, index alone for jvector and Lucene (their vectors are an in-memory list, or are added inside the Lucene build): the figure to compare ACROSS engines
for (arm, k), (v, meta) in (list(vals.get("build", {}).items()) if EXTRA else []):
    ing = vals.get("ingest", {}).get((arm, k))
    vals.setdefault("buildtotal", {})[(arm, k)] = (v + (ing[0] if ing else 0), meta)
arms = sorted({a for v in vals.values() for (a, _) in v})
pairs = [("jv-FP32", a) for a in arms if a.startswith("jv-") and a != "jv-FP32"] + [("ad-NONE", a) for a in arms if a.startswith("ad-") and a != "ad-NONE"] + [("jv-FP32", "ad-NONE")]
pairs += [(a, "ad-" + a[3:]) for a in arms if re.fullmatch(r"jv-PQ\d+", a) and "ad-" + a[3:] in arms]
pairs += [("lu-FP32", a) for a in arms if a.startswith("lu-") and a != "lu-FP32"] + [("jv-FP32", "lu-FP32")]
if dim: pairs += [(f"jv-PQ{dim // 8}", "lu-BBQ1"), (f"jv-PQ{dim // 4}", "lu-BBQ2")]
if "--main" in sys.argv: pairs += [("jv-BQ", "ad-BINARY")]   # main variant (L1MAIN, README "Main variant"): the same bucket (binary codes) inside and outside ArcadeDB; without --main the pairs are exactly as before
if "ad-NONE-SVG" in arms: pairs += [("jv-FP32", "ad-NONE-SVG")]   # storeVectorsInGraph arm (README "storeVectorsInGraph arm"): ad-NONE vs ad-NONE-SVG is already in the "ad-NONE vs each ad arm" rule; arms without it give exactly the old pairs
figs = []
for stat, v in sorted(vals.items()):
    for a, b in pairs:
        rows = [{"round": k, "arm": "A" if arm == a else "B", "value": s, **m} for (arm, k), (s, m) in sorted(v.items(), key=lambda x: x[0][1]) if arm in (a, b)]
        if any(r["arm"] == "A" for r in rows) and any(r["arm"] == "B" for r in rows):
            name = f"{d}/fig-{stat}-{a}-vs-{b}.jsonl"
            with open(name, "w") as o:
                for r in sorted(rows, key=lambda r: (r["round"], r["arm"])): o.write(json.dumps(r) + "\n")
            figs.append((name, stat))
if "--clean" in sys.argv:
    tools = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ab.py")   # rc.1 kit: the copy of tools/ab.py in this folder (SHA256SUMS)
    noise = os.environ.get("AB_NOISE", "/nonexistent-no-noise-log-on-this-host"); mins = []; out = []
    for j, stat in figs:
        r = subprocess.run([sys.executable, tools, j] + (["--noise", noise] if os.path.exists(noise) else []) + (["--higher-is-better"] if stat.startswith("qps") else []),
                           capture_output=True, text=True).stdout
        m = re.search(r"clean pairs (\d+)", r); mins.append(int(m.group(1)) if m else 0)
        out.append(f"== {os.path.basename(j)[4:-6]}\n" + "\n".join(l for l in r.splitlines() if re.search(r"clean pairs|VERDICT|median|interval|ratio|COLLISION|OTHER-BENCH", l)))
    print(f"MINCLEAN {min(mins) if mins else 0}"); print("\n".join(out))
