# Agent-memory benchmark

This page defines the benchmark that would show whether tiervec is a good index, or the best one, for the memory of an agent inside a database engine. **It is a definition. Most of it is not built, and it has no results.** The work is tracked in [#159](https://github.com/humemai/tiervec/issues/159) (the program) and [#160](https://github.com/humemai/tiervec/issues/160) (the other libraries). Why these questions and these baselines: [RELATED-WORK.md](RELATED-WORK.md).

Every size, rate, and bound on this page is a choice made here. A study fixes its own values in its rules before its first run, under the protocol of [ROADMAP.md](ROADMAP.md#how-to-read-measurements).

## The question

On a stream of creates, reads, updates, and deletes, does an index stay correct and keep its recall, how fast are searches while the writes run, what does a write cost, how much memory does a row take, and how soon does it answer after a restart, compared with the other designs a host could embed?

## What exists today

| Program | What it covers | What it lacks |
|---|---|---|
| `Bench` ([BENCHMARKS.md](BENCHMARKS.md)) | Static recall and latency, filtered cells with random filters, checks of writes, a concurrent phase of adds and deletes with recall scored once at its end, a JVector arm | Replacements under load, recall during the stream, real data in the concurrent phase, updates in the JVector arm |
| [`PacedWrites`](BENCHMARKS.md#paced-writes) | Adds at a fixed rate with paced searches, write amplification | Deletes and replacements |
| [`MergeGenerations`](BENCHMARKS.md#successive-merges) | Recall over successive merges with deletes | Writes during searches, other libraries |
| `AgentMemory`, the first step of [#159](https://github.com/humemai/tiervec/issues/159): [the runbook](#the-runbook) replayed into the exact, tier, live graph, and JVector arms ([how](#how-the-program-replays-a-runbook)) | Creates, updates, deletes, reuses, batches, searches, and checkpoints from a seed; violation counts; recall at every checkpoint; distance computations; rows built per row written; memory per row; the [screen](#the-screen), started by hand | Restarts, filtered searches, many collections as its own study, the Lucene arm, real embeddings, every timing |

## The workload

A **runbook** is a file of operations that is generated once from a seed and replayed unchanged by every arm. The [NeurIPS 2023 streaming track](https://arxiv.org/abs/2409.17424) uses the same idea.

| Operation | Meaning |
|---|---|
| `create` | Add a new id |
| `update` | Replace the vector of a live id |
| `delete` | Delete a live id |
| `reuse` | Delete an id and later add it again with another vector |
| `batch` | 1 to 50 creates, updates, and deletes that must become visible together |
| `search` | Top 10, without a filter |
| `filtered` | Top 10 among an allowed set |
| `checkpoint`, `restart` | Persist, then stop the process cleanly or kill it, and open again |

**Data.** Real text embeddings at 768 and 1536 dimensions, and a set at 1024 if one with a usable license is prepared. Generated data is for smoke runs only.

**Change over time.** The rows are clustered. Each round adds rows of one cluster, deletes rows of another, replaces rows, and expires the oldest ones, so the distribution moves. A run lasts until the live set has been replaced at least three times.

**Two mixes.** Read-heavy: four searches for each written row. Write-heavy: inserts, deletes, and searches near 4 : 4 : 1, as in the streaming track.

**Shapes.**

| Shape | Collections | Live rows each | Why |
|---|---|---|---|
| Personal memories | 100, 1,000, and 10,000 | 20, 200, and 2,000, and one skewed mix | One collection per user, agent, or thread |
| Large agent | 1 | 10,000 and 100,000 | A long-lived agent |
| Team memory | 1 | 1,000,000 | The target of [#60](https://github.com/humemai/tiervec/issues/60) |

**Concurrency.** 1 and 8 writer threads with 1 and 8 searcher threads. Writes and searches arrive at fixed offered rates, and a latency is counted from the scheduled start, so a stall is not hidden by the requests it delays.

**Filters.** Allowed shares 0.5, 0.1, 0.01, and 0.001, in three kinds: random, one that matches the cluster of the query, and one that matches a cluster far from the query.

## The runbook

`Runbook`, in `src/bench`, generates a runbook from a seed and writes it to one binary file that holds every vector, so that every arm reads the same bytes; the SHA-256 of the file goes into the results. A runbook has one or more collections of the same size, and each collection is replayed into its own index. Today the data is generated: unit vectors around random cluster centres, of two kinds, set by `--latent`.

| `--latent` | A row | A query |
|---|---|---|
| 0 | The centre of its cluster plus noise in all dimensions | Near a live row |
| 1 to the dimension | The centre of its cluster plus noise in the span of that many random directions, so the rows of a cluster use about that many dimensions | Drawn like a new row of the cluster of a random live row, so it is no near copy of a stored row |

- A collection starts with a load of its rows and a checkpoint. Then it runs in rounds. A round retires a tenth of the rows (at least 4): 30 percent expire (the oldest), 30 percent are rows of one cluster that are deleted, 30 percent are updated, and the rest are deleted and added again later under the same id (`reuse`). New rows replace the deleted ones.
- New and replaced vectors come from the cluster that grows in that round, so the distribution moves. The rounds end when as many row versions have been retired as the chosen turnover asks for.
- Part of the writes arrive in batches of 1 to 50, and no id appears twice in one batch. Searches are placed between the writes at a chosen number per written row, with the queries of the table above.
- A checkpoint carries the queries that recall is scored on. In this first version nothing is persisted at a checkpoint: restarts are a later step.

`--clusters` sets the clusters of a collection; by default there is one per 20 rows, at least 4 and at most 16. For every shape the program says how hard the queries of the checkpoints are, from the live rows alone and before any arm runs: the local intrinsic dimension (the maximum likelihood estimate from the 100 nearest rows, or all rows but one in a smaller collection) and the relative contrast (the mean distance to the live rows over the distance to the 10th nearest). Both use the Euclidean distance between unit vectors. The line is `DATA` in `result.txt`.

## Ground truth

At every checkpoint of the runbook the program computes the exact top 10 over the rows that are live at that point, with their current vectors, using its own scalar distance and none of the arm's kernels. The ground truth has its own test with known answers that include deletes and replacements. A score that differs from the exact one by more than 0.0001 counts as the score of another vector. The streaming track had to correct its published recall after the competition: the declared winner went from 0.985 to 0.0921 in the corrected table.

## How the program replays a runbook

```text
mkdir -p target/bench/classes
javac -d target/bench/classes --add-modules jdk.incubator.vector $(find src/main/java src/bench/java -name '*.java')
java -Xmx2g --add-modules jdk.incubator.vector -cp target/bench/classes ai.humem.tiervec.AgentMemory \
  --shapes smoke,p20 --variants 'exact exact; tier tier; graph-a12 graph graph.alpha=1.2' --mix read --out target/agent-memory/out
```

The arm `jvector` needs the jars of the profile `bench` of the pom. Fetch them, compile `src/bench/jvector` with the rest, and put them on the class path:

```text
mvn -B -ntp -Pbench dependency:copy-dependencies -DoutputDirectory=target/bench/lib -DincludeScope=runtime
javac -d target/bench/classes --add-modules jdk.incubator.vector -cp 'target/bench/lib/*' \
  $(find src/main/java src/bench/java src/bench/jvector -name '*.java')
java -Xmx2g --add-modules jdk.incubator.vector -Djvector.vectorization_provider=panama -cp 'target/bench/classes:target/bench/lib/*' \
  ai.humem.tiervec.AgentMemory --variants 'exact exact; jvector jvector' --out target/agent-memory/out
```

The program writes one runbook per shape into `--out`, reads it back, replays it into each variant, and writes there `result.txt` (one line per fact), `results.tsv` (one row per shape and variant), `curve.tsv` (recall at every checkpoint), `graph.tsv` ([the shape of the graphs](#the-shape-of-the-graphs) at every checkpoint), and `summary.md`. A variant is `name arm key=value ...`. The arm `tier` takes every key of `TieredIndex.Config` (see `Config.toMap`) except `buildThreads`, `tailAgeMs`, and `mergeRatePerSec`; the arm `graph` takes `graph.M`, `graph.efConstruction`, `graph.efSearch`, `graph.alpha`, `graph.sq8`, and `consolidateAt` (default 0.2: `consolidate()` runs when the deleted nodes are more than that share of the live ones); the arm `rebuild` takes the `graph.` keys; the arm `exact` takes none; the arm `jvector` takes `degree` (32, on the base layer), `upperDegree` (16, above it), `efConstruction` (200), `efSearch` (100), `alpha` (1.2), `neighborOverflow` (1.2), and `removeAt` (0.2). Its degrees and beam widths are those of the graphs of the tier, and `alpha` and the overflow are those of JVector's tuning page. For an arm that reports them, `result.txt` also has an `ABOUT` line (the library, its SIMD provider, and the settings) and a `COUNTERS` line (counts of the arm's own work, summed over the collections). The shapes are `smoke` (3,000 rows), `p20`, `p200`, and `p2000` (50, 20, and 5 collections of that many rows), and `a10k` and `a100k`. `--turnover` sets how many row versions are retired per live row (3 by default), and `--checkpoints` how many checkpoints follow the load (10 by default).

The arm `rebuild` is the control of a graph that lives through changes ([#161](https://github.com/humemai/tiervec/issues/161)): at every checkpoint it builds a new graph from the live rows, in the order of their last write, and answers the checkpoint from it. Between the checkpoints its searches are exact, so it says nothing about searches under writes, and its distances per written row hold the rebuilds of the checkpoints.

- **Nothing depends on timing.** The tier runs with one build thread, no seal by age, and no merge throttle, and with a host environment that runs each seal and merge inside the write that made it due. So one runbook gives the same segments on every run.
- **Every search is checked** against the live rows: a deleted or never-added id, a score of another vector, and fewer than `k` ids when `k` rows are live.
- **Writes are probed.** After a write, a search that allows only the written ids must see all of them. A filter of that size takes the exact path of every arm. The large shapes probe every 10th or 100th single write, and every batch.
- **Batches are watched.** On the tier a second thread searches while the batch is applied. On the arms without atomic batches the replay applies one write at a time and searches in between, so their `partOfBatch` count is expected and does not fail the run.
- **Rows built per row written** counts the rows put into a graph (for the tier by seals and merges, for the live graph once per added row) over the rows created, updated, or reused; for the JVector arm once per added row, and the repairs of a removal add no row. **Bytes held per row** is what `bytesHeld()` reports, over the live rows at the end; the JVector arm reports JVector's estimate of its graph, the float32 vectors that it still holds, and 12 bytes for each ordinal it has used. **Heap bytes per row** is what the heap gives back when the arms are closed and dropped, after full collections; for small collections the noise of the runtime is as large as the figure.

### The JVector arm

`JVectorMemoryArm`, in `src/bench/jvector`, drives one in-memory graph through JVector's public builder and searcher. The version is the one that the profile `bench` pins, a release candidate ([BENCHMARKS.md](BENCHMARKS.md#the-jvector-arm) says why).

- An update marks the old node with `markNodeDeleted` and then adds a node with the new vector. Two versions of an id are never visible together, and between the two calls the id is not visible. A delete only marks.
- `removeDeletedNodes` runs when the marked nodes are more than `removeAt` times the live ones. JVector says that the call is not thread-safe against other modifications, so writers wait while it runs. The arm counts the writes that waited (`writerWaits`) and the time they waited. The replay writes on one thread, so no write waits there; `JVectorMemoryArmCheck` checks the counted wait, and writers and searchers on several threads beside removals.
- There are no atomic batches.
- **No promise of `k` ids.** In the CI smoke, some searches of collections of 20 rows returned fewer ids than there were live rows (`tooFewHits`), while the 3,000-row shape had none. For the first such search that `JVectorMemoryArmCheck` meets in a churn of 20 rows, it prints what a walk of the base layer finds: in [one run](https://github.com/humemai/tiervec/actions/runs/46125658829), the entry node was live and had no neighbor there, with no node left marked, so the search reached 1 of the 20 live rows. How the graph gets into that state is not established. The arm therefore does not promise `k` ids: those searches are counted and do not fail the run, and they lower the recall.
- **Not the same on every run.** Two runs of one seed gave slightly different recall and distance counts on collections of 20 rows. By JVector's source, `removeDeletedNodes` picks neighbors at random, without a seed, for a node that has no replacement candidate, which is the likely reason.
- A search whose filter allows at most `efSearch` ids scores those ids exactly with JVector's similarity function. JVector has no exact path of its own.
- The arm does not call `cleanup()`, which marks every change as complete.
- The replay says nothing about the recall or the answers of searches that run while a removal runs.

## The screen

The screen is the quick round for one idea: small shapes, many named variants over the same runbooks, and only measures that do not depend on timing. The ideas, with the bar each one has to pass, are in [SCREEN-IDEAS.md](SCREEN-IDEAS.md). Its results pick what is worth a confirmation run at larger sizes; they change no default by themselves.

Start it by hand on any branch that has it. The result files are the artifact `screen-results` of the run, and the summary table is on the run page.

```text
gh workflow run screen.yml --ref <branch>
gh workflow run screen.yml --ref <branch> -f shapes=p20,a10k -f variants='tier tier; mine tier tailLimit=500' -f seed=2
```

`--preset screen` runs `p20`, `p200`, `p2000`, and `a10k` (add `a100k` with `--shapes`), 768 dimensions with `--latent` 768 and `--clusters` 8 ([how that data was chosen](SCREEN-IDEAS.md#the-data-of-the-screen)), the write-heavy mix, and the variants of `src/bench/agent-memory-screen.variants`. It reports violations, recall at every checkpoint and its mean over the checkpoints after the load (`recallChurnMean`), distance computations per search and per written row, rows built per row written, and memory per row.

- **Distance computations** are counted by a copy of the library in which `scripts/instrument_distances.py` adds one line to every distance kernel; the library that ships is unchanged, and the screen refuses a build that does not count. Compile `target/agent-memory/src` (the copy) instead of `src/main/java`, with `src/bench/jvector` and the jars as above, because the variants file has JVector variants. The JVector arm counts in the score functions that it gives JVector, one for each score that JVector asks for, in searches, inserts, and removals. A computation on quantized codes counts as one, and `quantizedPerSearch` says how many of them there were. Per search they are counted over the queries of the checkpoints, the same queries that recall is scored on; per written row over the writes and their maintenance. Probes are in neither.
- **The same numbers again.** On one machine, two runs of one seed on one commit give the same results apart from the heap column, and the CI smoke checks that. The JVector arm is the exception ([why](#the-jvector-arm)), and the check leaves its rows out. Another CPU can have other vector lanes, which can move a float32 score in its last digits.
- **What a screen cannot say.** The data is generated, the sizes are small, and nothing is timed. Recall is usually higher on small sets, and the size at which a graph beats a scan depends on the dimension and the machine.

### The shape of the graphs

For every arm with graphs (`tier`, over its sealed segments, `graph`, and `rebuild`), `graph.tsv` has one row per checkpoint, summed over the collections, about layer 0 only. A node is a live row of a graph: not deleted, and in a tier not hidden. Its degree counts only the entries of its list that lead to live nodes. `shareBelowM` is the share of nodes with a degree below `M`, which is half of the layer-0 limit. `listedByNone` counts the nodes that the list of no live node holds, and `unreached` the nodes that no walk from the entry reaches. `crossClusterShare` is the share of the edges between live nodes that lead to another cluster of the generated data, where the cluster of a row is its nearest cluster center. `listedPerNode` counts all entries, the deleted ones too. The numbers are read through reflection, and no distance of theirs is counted.

The copy of the library that counts distances also counts what the neighbor selection of `GraphIndex` does. For each variant, `result.txt` has a `PRUNE` line per caller (`insert`: the lists of a new node; `reverse`: the list of a neighbor that is full; `repair`: `consolidate()`) with the calls, their candidates, the candidates that the diversity rule examined before the list was full, the ones that it kept, the ones that the fill added, and the calls that stopped with candidates never examined.

## Metrics

| Metric | Definition | Unit |
|---|---|---|
| Violations | A write not seen by a later search, a deleted id returned, an old vector's score returned, part of a batch seen, or fewer than `k` exact-path hits when `k` live rows qualify | count, must be 0 |
| Filter shortfall | Approximate filtered searches that return fewer than `k` although `k` rows qualify | share of searches |
| Recall under churn | Recall@10 at each checkpoint, as a curve, with its minimum, and its mean over the checkpoints after the load | share |
| Search latency under writes | p50 and p99, and searches per second, while the writes run, at matched recall of 0.95 and 0.99 | µs, 1/s |
| Write latency | p50 and p99 until the write is visible to a search, refreshes included | µs |
| Write capacity | The highest write rate that is sustained without back pressure or a growing backlog | rows/s |
| Write cost | Process CPU time per 1,000 operations, rows built or rebuilt per row written, and bytes written to disk per row written | s, ratio, bytes |
| Memory per vector | Heap after garbage collection, resident bytes, and mapped bytes, divided by live rows, and the peak during background work | bytes |
| Many collections | Memory and threads per collection, and the first search of a collection that was closed | bytes, µs |
| Filtered search | Recall and latency per share and filter kind, and filter calls per search | share, µs |
| Restart | Time from process start to the first correct answer, after a clean checkpoint and after a kill with unpersisted writes | s |

Latencies of two arms are compared only at matched recall, found as `Bench` does it ([Latency at matched recall](BENCHMARKS.md#latency-at-matched-recall)).

## Arms

| Arm | What it is | Fairness notes |
|---|---|---|
| Exact | `FlatIndex`, a scan of every live row | The truth for small collections, and the arm to beat there |
| tiervec tier | Default settings, then each candidate as its own arm: merge policy, `INSERT` merge, SQ8, mapped vectors | Settings are named in the result file |
| Live graph | One `GraphIndex` with adds, deletes, and scheduled `consolidate()`, at `alpha` 1.0 and 1.2 | In memory and FreshVamana-like. It is not FreshDiskANN |
| JVector | One graph, concurrent adds, `markNodeDeleted`, `removeDeletedNodes` on a schedule | Writers wait during removal, because it is not thread-safe against them, and the wait counts. No atomic batches. Built: [the JVector arm](#the-jvector-arm) |
| Lucene | `IndexWriter` with deletes and updates by id, near-real-time refresh, float32 and quantized HNSW formats | A refresh before a search that must see a write counts in the write latency |

All arms run in the same process shape, on the same JDK, cores, heap limit, data, and runbook, one arm per fresh JVM. Each library is run with its documented settings for this use, and the build parameters are matched where they mean the same thing, as in [#121](https://github.com/humemai/tiervec/issues/121). An arm that lacks a guarantee (for example atomic batches) reports its violations, and the table of results says that it does not give that guarantee.

**Not in the same harness.** hnswlib, DiskANN, FreshDiskANN, and SPFresh are C++ or Rust systems, and the last three are built for data larger than memory. A comparison across processes and languages would measure the bindings too. If one is run, it is reported apart and labelled so. This review found no SPFresh-style index that runs on the JVM, and did not search widely for one. `IvfIndex` with its full retrain is not a substitute and is not claimed as one. Database servers and engine indexes are compared in MMDBBench, not here.

## Verdicts

- A run with a violation in an arm that promises the guarantee fails, and its figures are not read.
- Every metric gets its own verdict per shape, against the best other arm, with the bounds written before the run. Latency follows the rule form of [#121](https://github.com/humemai/tiervec/issues/121): paired ratios with an interval, and an A/A control.
- The word "best" is used only for one metric on one shape, when tiervec is ahead of every in-process arm under that rule. It is used without a qualifier only if tiervec is behind no arm on any metric of any shape, which no single study can show.

## What would disprove the design

| If the benchmark shows | Then |
|---|---|
| The live graph, JVector, or Lucene keeps recall and latency at least as well at less write cost on the personal and large shapes | Segments are the wrong default there |
| The exact arm wins at every personal size | No graph is needed for most collections, and the threshold moves up |
| Recall falls over rounds in `INSERT` mode or the live graph | Graph reuse stays off until the repair of [#161](https://github.com/humemai/tiervec/issues/161) holds |
| p99 grows with the batch rate | Batch visibility is redesigned ([#163](https://github.com/humemai/tiervec/issues/163)) |
| Memory per vector is above the other arms | Compressed codes move ahead of new features ([#164](https://github.com/humemai/tiervec/issues/164)) |

## Order of building

1. The runbook, the ground truth, and the violation counters, with the tier, the exact arm, and the live graph, at the smoke size in CI.
2. Restart and filter operations, then the many-collection shapes.
3. The JVector and Lucene arms.
4. The rules of the first timed study, written on its issue before any timed run.
