# Status

Experimental. The sources come from the first experiment and were imported unchanged except for the test wrappers and the small edits listed under "Import notes". Measurements of the experiment live in a private harness and are not published yet. This repository has no benchmark numbers.

## What exists

Each part has the test programs of the experiment, wrapped as JUnit 5 tests that CI runs on JDK 21 and 25 (see [README](../README.md#build-and-test)).

| Part | What exists | Tests |
|---|---|---|
| Flat tier | `FlatIndex` with `NONE`, `SQ8`, and `BINARY` scans, compaction, `serialize` and `load`, scalar twin kernels | Correctness program, threaded program (5 s fast, 20 s and 60 s slow), correctness again with the scalar twins |
| Graph tier | `GraphIndex` (concurrent HNSW, delete, `consolidate`, optional SQ8), `SealedGraph`, `seal` and `fromSealed` in memory, `SealedGraph.serialize` and `load` (one segment file, CRC32C per section) | Functional program (quick mode), threaded program with background consolidate (5 s fast, 60 s slow for fp32 and SQ8), latency during consolidate (slow), segment round trip and damage tests |
| IVF | `IvfIndex` as a standalone index: lazy training, background retrain with atomic swap, drift detection, compaction | Single-threaded program at n = 6000 (fast, vector and scalar) and n = 20000 (slow), drift program (slow), threaded program (5 s fast, 10 s slow) |
| Tier | `TieredIndex`: exact tail, sealed graph segments, merge, tombstones with a version floor, `rebuildFrom`, `checkpoint` and `open` for a settled tier (segments plus a manifest per generation, [format](SEGMENT-FORMAT.md)) | The tier program, one test at a time (fast), the 10 s threaded test (slow), reopen, manifest and crash-step tests (fast) |

The wrappers assert two things per program: zero failed checks, and at least the number of checks the program reported in earlier runs of the experiment. The program code itself is not changed in a way that weakens an assertion.

## What does not exist yet

- Serialization of IVF and of the flat tail. `FlatIndex` has its own `serialize` and `load`. The graph segments and a settled tier have the first cut ([SEGMENT-FORMAT.md](SEGMENT-FORMAT.md)): copy-based reads, no memory mapping, no tail replay, no non-empty tombstones (`checkpoint` refuses a tier with a tail, a floor entry, or hidden copies), every section verified at open. When `open` fails, the host falls back to `rebuildFrom`.
- Engine integration. Nothing here is wired into ArcadeDB or any other engine.
- IVF as a segment type of the tier. `IvfIndex` is standalone.
- A scalar twin for the graph kernels (`Kern`). `GraphIndex`, `SealedGraph`, and `TieredIndex` need the `jdk.incubator.vector` module at run time.
- A native-image test in CI.
- A release: no publish workflow, version `0.1.0-SNAPSHOT`.
- Published measurements. They are in a private harness and will be published later.
- The longer mode of the graph test program (a recall curve at n = 20000) is not wired into the JUnit tests.

## Import notes

- The Java package stays `com.humemai.vec` for now. A rename before the first release is a later decision.
- Comments and Javadoc in the sources still use experiment-internal names (requirement ids, fix numbers, rule names). They will be cleaned up when the API is settled.
- The graph test helper `T` was renamed `GraphT`, because the flat tests have their own `T` and both now live in one module and one package.
- The test helpers got an `exitOnFailure` flag (default true, so a program run alone still ends the JVM on failure), and the end of each program's `main` is guarded by it. A JUnit wrapper sets it to false and reads the counters instead, because a `System.exit` would end the surefire JVM.
- The micro-benchmark `FlatMicro`, the benchmark harness, and the native-image smoke test of the experiment are not part of this repository.
- One edit to a main source, `GraphIndex.java` (issue [#3](https://github.com/humemai/tiervec/issues/3), found by the slow CI job): `consolidate()`, `repairOrphans` and `seal()` size their bitsets and arrays from a snapshot of `nextOrd`, and a concurrent `add` can publish a node allocated after the snapshot as the entry point (or, in `repairOrphans`, return one from its search), so the node index fell past the end. `bit` now reads an index past the end as not set, `repairOrphans` sizes its arrays to cover the entry point, and `seal()` ignores an entry point from after its snapshot. `GraphConsolidateRaceTest` builds that view without threads. It was pushed alone first and failed the fast job on the unfixed code.
