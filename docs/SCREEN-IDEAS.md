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

A possible cause, read from the code and not measured: `GraphIndex` drops a candidate neighbour only when `alpha` times its distance to a kept neighbour is at most its distance to the node. If most candidates are about as far from each other as from the node, `alpha` 1.2 drops few of them, and a list fills with the nearest candidates, without the spread that leads a search across the graph. No data with a lower intrinsic dimension at these sizes was run. The result is about `alpha` in this graph, which does not have the repair rules of FreshDiskANN. It belongs to [#161](https://github.com/humemai/tiervec/issues/161).

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
