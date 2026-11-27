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

**The data of the screen from now on: 768 dimensions, `--latent` 768, `--clusters` 8**, the first candidate of step 2 that met the threshold. On `a10k` with seed 1 its queries have an intrinsic dimension of 103.4 and a relative contrast of 1.599. It is harder than real text embeddings are likely to be, so it stretches the designs and says little about the recall a host would see. The default tier stays close to 1 on it.

The lowest recall of `graph-a10` and `tier-m1` is at the first checkpoint, right after the load, which puts all rows into half of the clusters. A bar on `recallLowest` would so judge the load and not the churn. `results.tsv` has a new column, `recallChurnMean`, the mean recall of the ten checkpoints after the load, and the bars below use it.

## How to add an idea

1. Add a row below with its source, hypothesis, and pass bar, and the status `not run`. The bar names a shape, a column of `results.tsv`, and a number.
2. Add its variants to `src/bench/agent-memory-screen.variants`, or give them to the run. A change in the library is tried on its branch, as a new setting or a new arm.
3. Commit, then run the screen on that commit: `gh workflow run screen.yml --ref <branch>`.
4. Write the result and the link to the run into the row, and set the status to `passed` or `failed`. Change a bar only before the first run.

## Ideas

| Id | Idea and source | Hypothesis | Pass bar | Status and result |
|---|---|---|---|---|
| S1 | A live graph with `alpha` 1.2. [FreshDiskANN](https://arxiv.org/abs/2105.09613) reports stable recall under churn with its repair rules for the values of `alpha` it tried except 1; [#161](https://github.com/humemai/tiervec/issues/161) | Over three replacements of the live rows, `alpha` 1.2 keeps recall better than `alpha` 1.0 | On `a10k`, with each of the seeds 1, 2, and 3: `recallChurnMean` of `graph-a12` is at least 0.005 above that of `graph-a10`, with at most 1.25 times its `distancesPerSearch` | not run |
| S2 | Merge by insertion into the largest segment, `MergeMode.INSERT` ([#16](https://github.com/humemai/tiervec/issues/16), [#161](https://github.com/humemai/tiervec/issues/161)) | It keeps the recall of a rebuild and writes cost less graph work | On `a10k`, with each of the seeds 1, 2, and 3: `recallChurnMean` of `tier-m1-insert` is at most 0.005 below that of `tier-m1`, with at most 0.7 times its `distancesPerRowWritten` | not run |
| S3 | A live graph with chunks of 64 slots. On the first data, at 20 rows, the graph held about four times the bytes per row of the tier. The cause in the code: `GraphIndex` allocates a whole chunk of 1,024 slots (`CHUNK_SHIFT` 10) for its first node, with vectors, ids, link references, states, and in-degrees for every slot, and `bytesHeld` counts it all. At 768 dimensions one chunk is 3,166,224 bytes, or 158,311 bytes for each of 20 rows before any link list | The chunk is why a small live graph takes so much memory, and chunks of 64 slots remove most of it without changing the graph | With seed 1: `bytesHeldPerRow` of `graph-a10` with `CHUNK_SHIFT` 6 is at most 1.5 times that of `tier` on `p20` and on `p200`, and on every shape of the screen its `recallChurnMean`, `distancesPerSearch`, and `distancesPerRowWritten` equal those of a run with `CHUNK_SHIFT` 10 that reports the same vector lanes | not run |

The bars of S1 and S2 were changed on 2026-11-28, before their first run on `a10k`. They named `recallLowest` (see [above](#the-data-of-the-screen)), one seed, and for S2 the default tier. At the default `maxSegments` of 8 the count rule merges the smallest segments, so by the code the largest one, which `MergeMode.INSERT` needs, is not among them under churn. S2 is so judged with `maxSegments` 1, where every merge takes the largest segment; `tier` and `tier-insert` still run, to show what the default layout does.
