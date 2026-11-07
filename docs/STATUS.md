# Status

Experimental. The sources come from the first experiment and were imported unchanged except for the test wrappers and the small edits listed under "Import notes". Measurements of the experiment live in a private harness and are not published yet. This repository has no benchmark numbers.

## What exists

Each part has the test programs of the experiment, wrapped as JUnit 5 tests that CI runs on JDK 21 and 25 (see [README](../README.md#build-and-test)).

| Part | What exists | Tests |
|---|---|---|
| Flat tier | `FlatIndex` with `NONE`, `SQ8`, and `BINARY` scans, compaction, `serialize` and `load`, scalar twin kernels | Correctness program, threaded program (5 s fast, 20 s and 60 s slow), correctness again with the scalar twins |
| Graph tier | `GraphIndex` (concurrent HNSW, delete, `consolidate`, optional SQ8), `SealedGraph`, `seal` and `fromSealed` in memory, `SealedGraph.serialize` and `load` (one segment file, CRC32C per section) | Functional program (quick mode), threaded program with background consolidate (5 s fast, 60 s slow for fp32 and SQ8), latency during consolidate (slow), segment round trip and damage tests |
| IVF | `IvfIndex` as a standalone index: lazy training, background retrain with atomic swap, drift detection, compaction | Single-threaded program at n = 6000 (fast, vector and scalar) and n = 20000 (slow), drift program (slow), threaded program (5 s fast, 10 s slow) |
| Tier | `TieredIndex`: exact tail, sealed graph segments, merge, tombstones with a version floor, `rebuildFrom`, `checkpoint` and `open` for a settled tier (segments plus a manifest per generation, [format](SEGMENT-FORMAT.md)) | The tier program, one test at a time (fast), the 10 s threaded test (slow), reopen, manifest and crash-step tests (fast), the record-id reuse tests (see below) |

The wrappers assert two things per program: zero failed checks, and at least the number of checks the program reported in earlier runs of the experiment. The program code itself is not changed in a way that weakens an assertion.

## Record-id reuse and tombstones

A host database frees a record id and later gives it to a new record, so the tier sees `delete(id)` followed by `add(id, newVector)`. The version floor of `TieredIndex` keeps the old copies of an id hidden while the new vector is visible. Tests cover this ([#5](https://github.com/humemai/tiervec/issues/5)). They use a model that holds the current vector of every id and keeps the older vectors, so a resurrected vector is recognised by its score.

Cases, each on a fresh index:

| Case | What it does |
|---|---|
| `a_tail` | The old copy sits in the active tail: delete 30 percent, re-add 80 percent of them with new vectors, delete and re-add again, then seal |
| `a2_tail_and_segments` | The old copies sit in one segment and in the tail: delete, re-add, seal, merge |
| `b_sealed` | The old copies sit in sealed segments: delete, re-add (the new vector in the tail, the old one below the floor), seal beside the old copies, delete and re-add again, seal, merge |
| `c1_reuse_in_merge` | Re-add, and delete then re-add, inside a running merge of the segments that hold the old copies; then again after the merge |
| `c2_delete_in_merge` | Delete, delete then re-add, delete then re-add then delete, replace, and brand-new ids, all inside a running merge; no delete may be lost when the merged segment is swapped in |
| `e_floor_chain` | One set of ids goes through four generations, each in its own segment, so the floor moves up four times; then a merge |
| `s_seal_window` | Operations inside the publish window of a seal, with the frozen tail holding every old copy (before the swap) and right after the swap, using `setSealHooks` |
| `f_fuzz_background` | Slow variant only: the background worker seals and merges while one writer reuses, replaces and deletes ids and one reader searches all the time; a hit counts only for an id that was untouched for the whole search |

Every checkpoint searches plain, with the id alone allowed, and under `Freshness.SEALED_ONLY`, with the current vector and with an older vector of the id as the query. A deleted id, an older vector's score, a repeated id, a wrong score, or a reused id at rank 1 for its old vector fails the run. `size()` and `stats().live()` are compared with the model, and the floor map must be empty after a full merge in the deterministic cases. A merge window or seal window that is not reached fails the run, it is never skipped.

The tests are `RidReuseTest` (fast: n = 3000, dimension 32 and 128, two seeds, all deterministic cases, no reader thread), `RidReuseLongTest` and `RidReuseBigLongTest` (slow: all cases including the background case, with a concurrent reader that counts a hit only for an id untouched while the search ran; n = 3000 with three seeds on three configurations, and n = 20000), `RidReuseSwapRaceTest` (slow: a search paused inside its allow predicate while `mergeNow()` swaps and cleans up the floor), and `RidReuseMutantTest` and `RidReuseMutantLongTest`. The probe itself is `RidReuseProbe`.

Known issue: a search that overlaps a merge swap can return an id that was deleted before the search started ([#26](https://github.com/humemai/tiervec/issues/26)). The tests with a concurrent reader and `RidReuseSwapRaceTest` are slow for that reason, and they fail until the issue is fixed.

A probe that cannot fail is not a test, so the mutant tests check that it fails on a broken tier. The test reads `TieredIndex.java`, applies one single-line change (the anchor must occur exactly once), compiles the copy with the JDK compiler API into a temporary directory, and runs the probe in a class loader that prefers that directory. `TieredIndex.java` itself has no test hook. This needs a JDK, not a JRE: without a compiler the mutant tests are skipped with a message, and they fail when the `CI` environment variable is set.

| Mutant | Change | The probe must report |
|---|---|---|
| X0 | none (control) | a clean run |
| M1 | `add` removes the floor entry of the id instead of raising it | `staleHits` above 0, in `b_sealed` |
| M2 | `delete` takes a version but writes no floor entry | `lostDeletes` above 0, in `b_sealed` |
| M3 | `cleanupFloor` drops every entry after a merge | a wrong answer in `c2_delete_in_merge` |

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
