# Screen ideas

Every idea for the index is listed here before its first run, with its source, what it is expected to do, and the bar it has to pass. The result is added after the run. An idea that fails stays in the list with its numbers, so that it is not tried twice.

The ideas are tested in two stages ([plan on #159](https://github.com/humemai/tiervec/issues/159#issuecomment-7222822816)): many quick rounds on the [screen](AGENT-MEMORY-BENCHMARK.md#the-screen), at small sizes, and then a confirmation at larger sizes only for the ideas that pass, under rules written before that run. A screen has no timing, and its sizes and data are small and generated, so no default changes on a screen result alone.

## The data of the screen

The [first run](https://github.com/humemai/tiervec/actions/runs/46041361120) used 128 dimensions with noise in all of them and queries near stored rows. It ran the exact and tier arms only, and their lowest recall was 0.999 or more at every size, so a difference between two designs had no room to show.

Harder data is chosen once, by this rule, written before any run on it. Only the baseline arms run while it is chosen (`exact`, `tier`, and `graph-a10`, the live graph at `alpha` 1.0), never an idea.

- The candidates are 768 dimensions with `--latent` 32, 64, 128, 256, and 768, in this order ([the two kinds of data](AGENT-MEMORY-BENCHMARK.md#the-runbook)).
- The choice is the first candidate at which, on `a10k` with seed 1 and the write mix, `recallLowest` of both `tier` and `graph-a10` is at most 0.97. If none is, the choice is 768.
- The chosen data, its difficulty (the `DATA` lines), and the runs of every candidate that was tried are recorded here.

### Step 1: dimensions that the noise spans

All five candidates ran on `a10k`, seed 1, on commit `d84f2c6` ([#175](https://github.com/humemai/tiervec/pull/175)), with 0 violations. Intrinsic dimension and contrast are the `DATA` line.

| `--latent` | Intrinsic dimension | Contrast | `recallLowest` of `tier` | `recallLowest` of `graph-a10` | Run |
|---|---|---|---|---|---|
| 32 | 16.5 | 2.036 | 1.0000 | 0.9990 | [46042784692](https://github.com/humemai/tiervec/actions/runs/46042784692) |
| 64 | 24.3 | 1.868 | 1.0000 | 0.9980 | [46042789843](https://github.com/humemai/tiervec/actions/runs/46042789843) |
| 128 | 34.6 | 1.766 | 0.9990 | 0.9960 | [46042794754](https://github.com/humemai/tiervec/actions/runs/46042794754) |
| 256 | 47.2 | 1.710 | 0.9980 | 0.9870 | [46042798887](https://github.com/humemai/tiervec/actions/runs/46042798887) |
| 768 | 91.3 | 1.631 | 0.9970 | 0.9850 | [46042803864](https://github.com/humemai/tiervec/actions/runs/46042803864) |

No candidate reached the threshold, so by the rule `--latent` is 768. That still leaves the baselines little room, so a second step follows.

### Step 2: clusters, written before any run of it

- At 768 dimensions with `--latent` 768, the clusters of a collection (`--clusters`, 16 on `a10k` so far) take 8, 4, and 2, in this order: with fewer clusters, more rows are near each query. 2 is the least that lets the distribution move.
- Only baselines run: `exact`, `tier`, `graph-a10`, and `tier-m1` (`tier maxSegments=1`, where every seal rebuilds all rows into one graph segment: the arm that idea S2 will be compared with).
- The choice is the first candidate at which, on `a10k` with seed 1 and the write mix, `recallLowest` of both `graph-a10` and `tier-m1` is at most 0.97. If none is, the choice is 2. The default tier is reported and does not decide: it searches several segments with a full beam each, and in step 1 at `--latent` 768 it computed 6.3 times the distances per search of `graph-a10`.

All three ran on `a10k`, seed 1, on commit `d342987` ([#176](https://github.com/humemai/tiervec/pull/176)), with 0 violations.

| `--clusters` | Intrinsic dimension | Contrast | `recallLowest` of `graph-a10` | `recallLowest` of `tier-m1` | `recallLowest` of `tier` | Run |
|---|---|---|---|---|---|---|
| 8 | 103.4 | 1.599 | 0.9530 | 0.9530 | 0.9980 | [46045542559](https://github.com/humemai/tiervec/actions/runs/46045542559) |
| 4 | 114.1 | 1.507 | 0.8560 | 0.8560 | 0.9950 | [46045546229](https://github.com/humemai/tiervec/actions/runs/46045546229) |
| 2 | 124.0 | 1.315 | 0.7750 | 0.7750 | 0.9850 | [46045549908](https://github.com/humemai/tiervec/actions/runs/46045549908) |

**The data of the screen from now on: 768 dimensions, `--latent` 768, `--clusters` 8**, the first candidate of step 2 that met the threshold. On `a10k` with seed 1 its queries have an intrinsic dimension of 103.4 and a relative contrast of 1.599. It is harder than real text embeddings are likely to be, so it stretches the designs and says little about the recall a host would see. The default tier stays close to 1 on it. The other shapes, with seed 1 on the same data settings: intrinsic dimension 9.8 and contrast 0.935 on `p20` (from 19 neighbours), 7.4 and 1.517 on `p200`, and 68.1 and 1.571 on `p2000`.

The lowest recall of `graph-a10` and `tier-m1` is at the first checkpoint, right after the load, which puts all rows into half of the clusters. A bar on `recallLowest` would so judge the load and not the churn. `results.tsv` has a new column, `recallChurnMean`, the mean recall of the ten checkpoints after the load, and the bars below use it.

## When segments lose on a shape

This is the rule for the comparison with one graph that is updated in place ([#160](https://github.com/humemai/tiervec/issues/160)). It was written on 2026-11-28, before any run of a JVector or Lucene arm. The numbers of the live graph on `a10k` were already known then ([S1](#s1-results)).

The tier here is the variant `tier`, its default settings. The challengers are `graph-a10` and `graph-a12` (the live graph), `jvector` and `jvector-a10`, and the two Lucene variants once their arm exists, `lucene` and `lucene-int8`. Tier and challengers run in the screen preset on its data above, in the same run and on one commit, with each of the seeds 1, 2, and 3.

- The rule judges a shape only if `tier` built graphs on it (`rowsBuiltPerRowWritten` above 0). Where it built none, the exact tail answers, and there are no segments to judge.
- Segments lose to a challenger on a shape if, with each of the three seeds, the challenger has 0 violations that fail a run and no search with fewer than `k` ids (`tooFewHits`, which does not fail the run of the JVector arm; this condition was added on 2026-11-28, after a CI smoke on other data showed such searches, and before any run of the arm on the data of the screen), a `recallChurnMean` that is at most 0.005 below that of `tier`, a `distancesPerSearch` that is at most that of `tier`, and a `distancesPerRowWritten` that is below that of `tier`.
- Segments lose on a shape if they lose to at least one challenger there. Every other outcome is reported as its numbers: a challenger can be better in one column and worse in another, and this rule does not rank that.
- A distance computation is not a time. A computation on quantized codes counts as one, each library runs its own kernels, and the arms run at fixed settings, not at matched recall. The write cost of Lucene includes the refresh that the replay asks for before every search, which on a shape that probes every write is one refresh per write.
- A shape that segments lose goes to a timed run, under rules written before that run. A screen changes no default.

## How to add an idea

1. Add a row below with its source, hypothesis, and pass bar, and the status `not run`. The bar names a shape, a column of `results.tsv`, and a number.
2. Add its variants to `src/bench/agent-memory-screen.variants`, or give them to the run. A change in the library is tried on its branch, as a new setting or a new arm.
3. Commit, then run the screen on that commit: `gh workflow run screen.yml --ref <branch>`.
4. Write the result and the link to the run into the row, and set the status to `passed` or `failed`. Change a bar only before the first run.

## Ideas

| Id | Idea and source | Hypothesis | Pass bar | Status and result |
|---|---|---|---|---|
| S1 | A live graph with `alpha` 1.2. [FreshDiskANN](https://arxiv.org/abs/2105.09613) reports stable recall under churn with its repair rules for the values of `alpha` it tried except 1; [#161](https://github.com/humemai/tiervec/issues/161) | Over three replacements of the live rows, `alpha` 1.2 keeps recall better than `alpha` 1.0 | On `a10k`, with each of the seeds 1, 2, and 3: `recallChurnMean` of `graph-a12` is at least 0.005 above that of `graph-a10`, with at most 1.25 times its `distancesPerSearch` | failed: `recallChurnMean` 0.023 to 0.059 lower with `alpha` 1.2 ([results](#s1-results)) |
| S2 | Merge by insertion into the largest segment, `MergeMode.INSERT` ([#16](https://github.com/humemai/tiervec/issues/16), [#161](https://github.com/humemai/tiervec/issues/161)) | It keeps the recall of a rebuild and writes cost less graph work | On `a10k`, with each of the seeds 1, 2, and 3: `recallChurnMean` of `tier-m1-insert` is at most 0.005 below that of `tier-m1`, with at most 0.7 times its `distancesPerRowWritten` | failed: the recall bar was missed with seed 1 (0.0103 lower); the cost bar held with every seed ([results](#s2-results)) |
| S3 | A live graph with chunks of 64 slots. On the first data, at 20 rows, the graph held about four times the bytes per row of the tier. The cause in the code: `GraphIndex` allocates a whole chunk of 1,024 slots (`CHUNK_SHIFT` 10) for its first node, with vectors, ids, link references, states, and in-degrees for every slot, and `bytesHeld` counts it all. At 768 dimensions one chunk is 3,166,224 bytes, or 158,311 bytes for each of 20 rows before any link list | The chunk is why a small live graph takes so much memory, and chunks of 64 slots remove most of it without changing the graph | With seed 1: `bytesHeldPerRow` of `graph-a10` with `CHUNK_SHIFT` 6 is at most 1.5 times that of `tier` on `p20` and on `p200`, and on every shape of the screen its `recallChurnMean`, `distancesPerSearch`, and `distancesPerRowWritten` equal those of a run with `CHUNK_SHIFT` 10 that reports the same vector lanes | passed: 0.30 and 0.23 times the tier on `p20` and `p200`, the other columns equal ([results](#s3-results)); not merged |
| S4 | `alpha` above 1 in two passes over the candidates, as `occlude_list` of [DiskANN's code](https://github.com/microsoft/DiskANN/blob/cpp_main/src/index.cpp) does: the first pass keeps what the rule keeps at `alpha` 1, and only the room that is left goes to candidates that pass at the configured `alpha`. At `alpha` 1.0 nothing changes. [#161](https://github.com/humemai/tiervec/issues/161) | One pass at `alpha` 1.2 fills a list with the nearest candidates before it looks at the far ones ([measured](#why-alpha-12-failed)). Two passes keep the far ones, so the layer-0 graph stays connected under churn and recall at `alpha` 1.2 is no lower than at 1.0 | On the data of the screen, with each of the seeds 1, 2, and 3: on `a10k` and on `p2000`, `recallChurnMean` of `graph-a12` with the change is at least that of `graph-a10` minus 0.002, with at most 1.25 times its `distancesPerSearch`; and on `a10k`, `unreached` of `graph-a12` in `graph.tsv` is at most 10 at every checkpoint | not run: the change is on the branch [`screen-two-pass-alpha-161`](https://github.com/humemai/tiervec/tree/screen-two-pass-alpha-161) (commit `f64a492`), not compiled yet |
| S5 | Repair of low-degree nodes in `consolidate()`, after Lucene's [GITHUB#16552](https://github.com/apache/lucene/pull/16559): there, a node that lost a neighbor and fell below half of its limit searches from its neighbors, adds the candidates that the rule keeps beside them, and gets their reverse links. Here the loss is not asked for, because `consolidate()` already refills a list that lost a neighbor, and the low degrees measured here come from insertion: every live node with fewer than `M` layer-0 entries searches (beam `efConstruction`) and is repaired the same way | 21.6% of the rows of `graph-a10` have fewer than `M` live neighbors after the load. More ways in and out of them keep recall closer to a rebuild's over a long churn | The [long churn](#the-long-churn) with each of the seeds 1, 2, and 3: `recallChurnMean` of `graph-a10` with the repair is at least 0.005 above that of `graph-a10` without it, with at most 1.25 times its `distancesPerRowWritten` | not run: not built yet |
| S6 | The [long churn](#the-long-churn) of the live graph and of the insert merge against a rebuild of the same live rows, the test that [#161](https://github.com/humemai/tiervec/issues/161) asks for. [FreshDiskANN](https://arxiv.org/abs/2105.09613) reports falling recall and degree over cycles at `alpha` 1; Lucene measured its decay over [100 generations](https://github.com/apache/lucene/issues/16552) | A graph that lives through 15 turnovers keeps the recall of a rebuild | On `a10k` with `--turnover 15 --checkpoints 15`, with each of the seeds 1, 2, and 3: for `graph-a10` against `rebuild`, and for `tier-m1-insert` against `tier-m1`, `recallChurnMean` is at least the control's minus 0.005 and `recallFinal` at least the control's minus 0.01; and `unreached` in `graph.tsv` is at most 10, 0.1% of the live rows, at every checkpoint | started on commit `7ab31fd`, one run per seed, results not read yet: [46123693234](https://github.com/humemai/tiervec/actions/runs/46123693234), [46123694997](https://github.com/humemai/tiervec/actions/runs/46123694997), [46123696956](https://github.com/humemai/tiervec/actions/runs/46123696956) |

### The long churn

`--turnover 15 --checkpoints 15` on `a10k`: 150 rounds, each retiring a tenth of the rows, with a checkpoint every 10 rounds, on the data and the write mix of the screen. The rows that arrive come from one cluster after another, so the distribution goes around the 8 clusters many times. The control of the live graph is the arm `rebuild`, a new graph from the live rows at every checkpoint; the control of the insert merge is `tier-m1`, whose every merge rebuilds the same layout. The variants are `graph-a10`, `graph-a12`, `rebuild`, `tier-m1`, and `tier-m1-insert`. The bars of S4, S5, and S6 were written on 2026-11-28, before any run of them.

The bars of S1 and S2 were changed on 2026-11-28, before their first run on `a10k`. They named `recallLowest` (see [above](#the-data-of-the-screen)), one seed, and for S2 the default tier. At the default `maxSegments` of 8 the count rule merges the smallest segments, so by the code the largest one, which `MergeMode.INSERT` needs, is not among them under churn. S2 is so judged with `maxSegments` 1, where every merge takes the largest segment; `tier` and `tier-insert` still run, to show what the default layout does.

## Results

All runs are the screen preset on its data above, and none has a violation. Nothing is timed. Every variant and shape ran on commit `9754ef8` ([#177](https://github.com/humemai/tiervec/pull/177)), the commit with the bars, once for each of the seeds 1, 2, and 3.

### S1 results

`alpha` 1.2 failed: with every seed it lost recall. On `a10k`:

| Seed | `recallChurnMean` `graph-a10` | `recallChurnMean` `graph-a12` | Difference (needs +0.005 or more) | `distancesPerSearch` ratio (at most 1.25) | Holds | Run |
|---|---|---|---|---|---|---|
| 1 | 0.9747 | 0.9176 | -0.0571 | 0.828 | no | [46047276298](https://github.com/humemai/tiervec/actions/runs/46047276298) |
| 2 | 0.9741 | 0.9508 | -0.0233 | 0.839 | no | [46047279946](https://github.com/humemai/tiervec/actions/runs/46047279946) |
| 3 | 0.9733 | 0.9140 | -0.0593 | 0.820 | no | [46047284166](https://github.com/humemai/tiervec/actions/runs/46047284166) |

It also computed 2.0 to 2.1 times the distances per written row. The gap was larger on `p2000`: `recallChurnMean` 0.7694, 0.7565, and 0.7653 for `graph-a12` against 0.9994, 0.9994, and 0.9997 for `graph-a10` with the seeds 1, 2, and 3.

The cause was measured afterwards ([next section](#why-alpha-12-failed)). No data with a lower intrinsic dimension at these sizes was run. The result is about `alpha` in this graph, which does not have the repair rules of FreshDiskANN. It belongs to [#161](https://github.com/humemai/tiervec/issues/161).

### Why alpha 1.2 failed

`GraphIndex` takes the candidates of a list in increasing distance and drops one only when `alpha` times its distance to a kept neighbor is at most its distance to the node. It does this in one pass and stops when the list is full. On this data the candidates of a node are nearly all about as far from each other as from the node, so at `alpha` 1.2 the rule drops almost none, the list is full before the far candidates are looked at, and the far candidates are the ones that lead to other clusters. The layer-0 graph then comes apart under churn. Low out-degrees are not the cause: they are no more common at `alpha` 1.2.

Measured on `a10k` with seed 1, on commit `79199be`, run [46123331066](https://github.com/humemai/tiervec/actions/runs/46123331066), with the [shape of the graphs](AGENT-MEMORY-BENCHMARK.md#the-shape-of-the-graphs). The recall of both variants equals that of S1, so the counting changed nothing.

| Checkpoint | Variant | Recall | Degree, median | Degree, 10th percentile | Share below `M` | Cross-cluster share | Live rows listed by no live row | Live rows unreached at layer 0 |
|---|---|---|---|---|---|---|---|---|
| 0 | `graph-a10` | 0.9530 | 21 | 10 | 0.2164 | 0.0037 | 0 | 0 |
| 0 | `graph-a12` | 0.9330 | 19 | 16 | 0.0005 | 0.0003 | 0 | 0 |
| 1 | `graph-a10` | 0.9750 | 28 | 16 | 0.0969 | 0.0044 | 7 | 0 |
| 1 | `graph-a12` | 0.9260 | 28 | 17 | 0.0330 | 0.0006 | 253 | 6147 |
| 4 | `graph-a10` | 0.9790 | 25 | 14 | 0.1280 | 0.0027 | 51 | 0 |
| 4 | `graph-a12` | 0.9420 | 25 | 16 | 0.0697 | 0.0010 | 354 | 5090 |
| 7 | `graph-a10` | 0.9710 | 29 | 16 | 0.0876 | 0.0012 | 21 | 2 |
| 7 | `graph-a12` | 0.8640 | 29 | 18 | 0.0319 | 0.0004 | 722 | 6363 |
| 10 | `graph-a10` | 0.9830 | 25 | 14 | 0.1386 | 0.0006 | 33 | 0 |
| 10 | `graph-a12` | 0.9140 | 26 | 16 | 0.0864 | 0.0027 | 672 | 8215 |

| Caller | Variant | Selections | Stopped at the limit before the last candidate | Candidates examined | Kept entries that came from the fill |
|---|---|---|---|---|---|
| insert | `graph-a10` | 42,687 | 51.7% | 62.1% | 0.0% |
| insert | `graph-a12` | 42,687 | 98.1% | 9.3% | 0.0% |
| reverse | `graph-a10` | 303,345 | 15.1% | 99.5% | 33.5% |
| reverse | `graph-a12` | 479,210 | 99.6% | 96.9% | 0.1% |
| repair | `graph-a10` | 127,701 | 2.4% | 98.7% | 54.3% |
| repair | `graph-a12` | 131,066 | 99.1% | 26.8% | 0.2% |

An unreached row can still be found, because a search starts on layer 0 at the node that the upper layers lead to. Read from the code and not measured: `consolidate()` links an unreached row only when a search through those upper layers finds a row that the walk from the entry reaches, so it does not join a part that the upper layers lead into. At `alpha` 1.0 the graph stayed in one piece in these three turnovers, but its share of cross-cluster edges fell from 0.0044 at checkpoint 1 to 0.0006 at checkpoint 10; the [long churn](#the-long-churn) follows it further.

### S2 results

| Seed | `recallChurnMean` `tier-m1` | `recallChurnMean` `tier-m1-insert` | Difference (needs -0.005 or more) | `distancesPerRowWritten` ratio (at most 0.7) | Holds | Run |
|---|---|---|---|---|---|---|
| 1 | 0.9851 | 0.9748 | -0.0103 | 0.528 | no | [46047276298](https://github.com/humemai/tiervec/actions/runs/46047276298) |
| 2 | 0.9837 | 0.9796 | -0.0041 | 0.529 | yes | [46047279946](https://github.com/humemai/tiervec/actions/runs/46047279946) |
| 3 | 0.9821 | 0.9788 | -0.0033 | 0.523 | yes | [46047284166](https://github.com/humemai/tiervec/actions/runs/46047284166) |

Insertion cost 0.52 to 0.53 times the distances of a rebuild per written row, so the cost bar held. The recall bar failed with seed 1, so the idea failed. With all three seeds insertion ended below the rebuild, by 0.0033 to 0.0103.

At the default `maxSegments` of 8, `tier-insert` gave the same number as `tier` in every column but the heap, and the same curve, on every shape and seed: `MergeMode.INSERT` changed nothing there.

### S3 results

The change is `CHUNK_SHIFT` from 10 to 6 in `GraphIndex.java` (commit `570cfcc`, on a branch that is not merged). `tier` and `graph-a10` ran with seed 1 on two runners with 8 lanes: with 1,024 slots on `9754ef8` ([46048690710](https://github.com/humemai/tiervec/actions/runs/46048690710)), with 64 slots on `570cfcc` ([46048694959](https://github.com/humemai/tiervec/actions/runs/46048694959)). An earlier run of `570cfcc` landed on 16 lanes ([46047308966](https://github.com/humemai/tiervec/actions/runs/46047308966)) and is not used for the bar; it held the same bytes per row.

| Shape | `bytesHeldPerRow` `tier` | `graph-a10`, 1,024 slots | `graph-a10`, 64 slots | 64 slots over `tier` | `heapBytesPerRow` `tier` | 1,024 slots | 64 slots |
|---|---|---|---|---|---|---|---|
| `p20` | 33,621 | 158,512 | 10,096 | 0.30 | 39149 | 159526 | 11110 |
| `p200` | 18,049 | 16,079 | 4,206 | 0.23 | 18744 | 16183 | 4311 |
| `p2000` | 6,510 | 5,009 | 4,020 | 0.62 | 6578 | 5022 | 4066 |
| `a10k` | 4,403 | 4,067 | 3,988 | 0.91 | 5670 | 4071 | 3994 |

On every shape `recallChurnMean`, `distancesPerSearch`, `distancesPerRowWritten`, and the recall curve of `graph-a10` were equal in the two runs, so the bar was passed. The factor of about four is a matter of small collections: at 200 rows the graph with 1,024 slots already held less than the tier. A pass here changes no default. The screen runs none of the library's tests and times nothing, so the change needs both before it can ship.
