#!/usr/bin/env python3
"""Copy the library sources and make the distance kernels count their calls.

The agent-memory benchmark (docs/AGENT-MEMORY-BENCHMARK.md, issue #159) reports the distance computations per search. The library has no
such counter, and adding one to its kernels would change the code that ships. So the benchmark compiles this copy: the sources of SRC with
one line added at the top of each distance kernel of Kern.java and Kernels.java, which adds to ai.humem.tiervec.DistanceCount (src/bench).
A dot product of a vector with itself is a norm and is not counted. The script fails when a kernel is not there exactly once, and when the
library calls a kernel method that the script does not know, so that a count cannot go wrong without notice.

It also makes the neighbor selection of GraphIndex.java count what it examines and keeps, into ai.humem.tiervec.PruneCount (src/bench): one
call before the fill of select(), and the caller named at each of its three call sites. A line that is not there exactly once fails the script.

usage: instrument_distances.py SRC OUT
    SRC is src/main/java. OUT is a new directory, or one that an earlier run of this script made, which is replaced.
"""

import re
import shutil
import sys
from pathlib import Path

MARKER = ".instrumented-copy"
NOT_A_NORM = "if (a != b || ao != bo) "

# file name -> the first line of a kernel -> the line that goes under it
KERNELS = {
    "Kern.java": {
        "static float dot(float[] a, int ao, float[] b, int bo, int n) {": NOT_A_NORM + "DistanceCount.full(1);",
        "static float l2(float[] a, int ao, float[] b, int bo, int n) {": "DistanceCount.full(1);",
        "static float dotQ8(float[] q, byte[] codes, int co, int n) {": "DistanceCount.quantized(1);",
    },
    "Kernels.java": {
        "static float dot(float[] a, int ao, float[] b, int bo, int n) {": NOT_A_NORM + "DistanceCount.full(1);",
        "static float l2sq(float[] a, int ao, float[] b, int bo, int n) {": "DistanceCount.full(1);",
        "static int dotI8(byte[] a, int ao, byte[] b, int bo, int n) {": "DistanceCount.quantized(1);",
        "static float dotF8(float[] q, int qo, byte[] c, int co, int n) {": "DistanceCount.quantized(1);",
        "static float dotF8x4(float[] q, int qo, byte[] c, int co, int n) {": "DistanceCount.quantized(1);",
        "static void block16(float[] q, int qo, byte[] c, int base, int n, float[] out) {": "DistanceCount.quantized(16);",
        "static int hamming(long[] a, int ao, long[] b, int bo, int words) {": "DistanceCount.quantized(1);",
    },
}
# The neighbor selection of GraphIndex.java, for ai.humem.tiervec.PruneCount (src/bench, issue #161): the line -> what replaces it.
SELECTION = {
    "int k = select(s, s.cn, s.cd, m, M, false, s.sel);": "PruneCount.site(0); int k = select(s, s.cn, s.cd, m, M, false, s.sel);",
    "int m = select(s, s.cn, s.cd, n, maxDeg, true, s.sel);": "PruneCount.site(1); int m = select(s, s.cn, s.cd, n, maxDeg, true, s.sel);",
    "else m = select(s, s.cn, s.cd, n, maxDeg, true, s.sel);": "else { PruneCount.site(2); m = select(s, s.cn, s.cd, n, maxDeg, true, s.sel); }",
    "if (fill) for (int i = 0; i < n && m < maxM; i++) if (!s.taken[i]) out[m++] = cn[i];": (
        "PruneCount.selected(s.taken, s.later, n, m, maxM, fill); if (fill) for (int i = 0; i < n && m < maxM; i++) if (!s.taken[i]) out[m++] = cn[i];"
    ),
}
# methods of Kern and Kernels that the rest of the library may call and that are not distances
NOT_DISTANCES = {"sum", "lanes", "normalized"}
# the files below the two counted ones: no other file of the library may call them
BACKENDS = ("NKernels.java", "VKernels.java", "VKern.java")


def declared_name(first_line: str) -> str:
    return first_line.split("(")[0].split()[-1]


def check_calls(sources: list[Path]) -> None:
    counted = {declared_name(line) for kernels in KERNELS.values() for line in kernels}
    call = re.compile(r"\b(Kern|Kernels|NKernels|VKernels|VKern)\.([a-zA-Z0-9_]+)\(")
    for path in sources:
        if path.name in KERNELS or path.name in BACKENDS:
            continue
        for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if text.lstrip().startswith(("*", "//", "/*")):
                continue
            for owner, method in call.findall(text):
                if owner in ("Kern", "Kernels") and method in counted | NOT_DISTANCES:
                    continue
                sys.exit(f"{path}:{number}: the call {owner}.{method} is not known to the distance counter")


def instrument(path: Path, kernels: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").split("\n")
    for first_line, added in kernels.items():
        at = [i for i, text in enumerate(lines) if text.strip() == first_line]
        if len(at) != 1:
            sys.exit(f"{path.name}: expected the line {first_line!r} once, found it {len(at)} times")
        lines.insert(at[0] + 1, "        " + added)
    path.write_text("\n".join(lines), encoding="utf-8")


def instrument_selection(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").split("\n")
    for line, replacement in SELECTION.items():
        at = [i for i, text in enumerate(lines) if text.strip() == line]
        if len(at) != 1:
            sys.exit(f"{path.name}: expected the line {line!r} once, found it {len(at)} times")
        indent = lines[at[0]][: len(lines[at[0]]) - len(lines[at[0]].lstrip())]
        lines[at[0]] = indent + replacement
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        sys.exit(__doc__)
    source, out = Path(argv[1]).resolve(), Path(argv[2]).resolve()
    if not source.is_dir():
        sys.exit(f"{source} is not a directory")
    if out == source or source in out.parents or out in source.parents:
        sys.exit("OUT and SRC must not contain each other")
    check_calls(sorted(source.rglob("*.java")))
    if out.exists():
        if not (out / MARKER).is_file():
            sys.exit(f"{out} exists and was not made by this script")
        shutil.rmtree(out)
    shutil.copytree(source, out)
    (out / MARKER).write_text("made by scripts/instrument_distances.py\n", encoding="utf-8")
    for name, kernels in KERNELS.items():
        found = sorted(out.rglob(name))
        if len(found) != 1:
            sys.exit(f"expected one file {name} in {source}, found {len(found)}")
        instrument(found[0], kernels)
    graph = sorted(out.rglob("GraphIndex.java"))
    if len(graph) != 1:
        sys.exit(f"expected one file GraphIndex.java in {source}, found {len(graph)}")
    instrument_selection(graph[0])
    print(f"instrumented {sum(len(k) for k in KERNELS.values())} kernels and the neighbor selection in {out}")


if __name__ == "__main__":
    main(sys.argv)
