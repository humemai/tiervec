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
| `AgentMemory`, the first step of [#159](https://github.com/humemai/tiervec/issues/159): [the runbook](#the-runbook) replayed into the exact, tier, and live graph arms ([how](#how-the-program-replays-a-runbook)) | Creates, updates, deletes, reuses, batches, searches, and checkpoints from a seed; violation counts; recall at every checkpoint; rows built per row written; bytes held per row | Restarts, filtered searches, many collections as its own study, the JVector and Lucene arms, real embeddings, every timing |

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

`Runbook`, in `src/bench`, generates a runbook from a seed and writes it to one binary file that holds every vector, so that every arm reads the same bytes; the SHA-256 of the file goes into the results. A runbook has one or more collections of the same size, and each collection is replayed into its own index. Today the data is generated: unit vectors around random cluster centres.

- A collection starts with a load of its rows and a checkpoint. Then it runs in rounds. A round retires a tenth of the rows (at least 4): 30 percent expire (the oldest), 30 percent are rows of one cluster that are deleted, 30 percent are updated, and the rest are deleted and added again later under the same id (`reuse`). New rows replace the deleted ones.
- New and replaced vectors come from the cluster that grows in that round, so the distribution moves. The rounds end when as many row versions have been retired as the chosen turnover asks for.
- Part of the writes arrive in batches of 1 to 50, and no id appears twice in one batch. Searches are placed between the writes at a chosen number per written row, each with a query near a live row.
- A checkpoint carries the queries that recall is scored on. In this first version nothing is persisted at a checkpoint: restarts are a later step.

## Ground truth

At every checkpoint of the runbook the program computes the exact top 10 over the rows that are live at that point, with their current vectors, using its own scalar distance and none of the arm's kernels. The ground truth has its own test with known answers that include deletes and replacements. A score that differs from the exact one by more than 0.0001 counts as the score of another vector. The streaming track had to correct its published recall after the competition: the declared winner went from 0.985 to 0.0921 in the corrected table.

## How the program replays a runbook

```text
mkdir -p target/bench/classes
javac -d target/bench/classes --add-modules jdk.incubator.vector $(find src/main/java src/bench/java -name '*.java')
java -Xmx2g --add-modules jdk.incubator.vector -cp target/bench/classes ai.humem.tiervec.AgentMemory \
  --shapes smoke,p20 --variants 'exact exact; tier tier; graph-a12 graph graph.alpha=1.2' --mix read --out target/agent-memory/out
```

The program writes one runbook per shape into `--out`, reads it back, replays it into each variant, and writes `result.txt` there. A variant is `name arm key=value ...`. The arm `tier` takes every key of `TieredIndex.Config` (see `Config.toMap`) except `buildThreads`, `tailAgeMs`, and `mergeRatePerSec`; the arm `graph` takes `graph.M`, `graph.efConstruction`, `graph.efSearch`, `graph.alpha`, `graph.sq8`, and `consolidateAt` (default 0.2: `consolidate()` runs when the deleted nodes are more than that share of the live ones); the arm `exact` takes none. The shapes are `smoke` (3,000 rows), `p20`, `p200`, and `p2000` (50, 20, and 5 collections of that many rows), and `a10k` and `a100k`.

- **Nothing depends on timing.** The tier runs with one build thread, no seal by age, and no merge throttle, and with a host environment that runs each seal and merge inside the write that made it due. So one runbook gives the same segments on every run.
- **Every search is checked** against the live rows: a deleted or never-added id, a score of another vector, and fewer than `k` ids when `k` rows are live.
- **Writes are probed.** After a write, a search that allows only the written ids must see all of them. A filter of that size takes the exact path of every arm. The large shapes probe every 10th or 100th single write, and every batch.
- **Batches are watched.** On the tier a second thread searches while the batch is applied. On the arms without atomic batches the replay applies one write at a time and searches in between, so their `partOfBatch` count is expected and does not fail the run.
- **Rows built per row written** counts the rows put into a graph (for the tier by seals and merges, for the live graph once per added row) over the rows created, updated, or reused. **Bytes held per row** is what `bytesHeld()` reports, over the live rows at the end.

## Metrics

| Metric | Definition | Unit |
|---|---|---|
| Violations | A write not seen by a later search, a deleted id returned, an old vector's score returned, part of a batch seen, or fewer than `k` exact-path hits when `k` live rows qualify | count, must be 0 |
| Filter shortfall | Approximate filtered searches that return fewer than `k` although `k` rows qualify | share of searches |
| Recall under churn | Recall@10 at each checkpoint, as a curve, with its minimum | share |
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
| JVector | One graph, concurrent adds, `markNodeDeleted`, `removeDeletedNodes` on a schedule | Writers wait during removal, because it is not thread-safe against them, and the wait counts |
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
