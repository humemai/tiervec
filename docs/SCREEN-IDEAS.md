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

## How to add an idea

1. Add a row below with its source, hypothesis, and pass bar, and the status `not run`. The bar names a shape, a column of `results.tsv`, and a number.
2. Add its variants to `src/bench/agent-memory-screen.variants`, or give them to the run. A change in the library is tried on its branch, as a new setting or a new arm.
3. Commit, then run the screen on that commit: `gh workflow run screen.yml --ref <branch>`.
4. Write the result and the link to the run into the row, and set the status to `passed` or `failed`. Change a bar only before the first run.

## Ideas

| Id | Idea and source | Hypothesis | Pass bar | Status and result |
|---|---|---|---|---|
| S1 | A live graph with `alpha` 1.2. [FreshDiskANN](https://arxiv.org/abs/2105.09613) reports stable recall under churn with its repair rules for the values of `alpha` it tried except 1; [#161](https://github.com/humemai/tiervec/issues/161) | Over three replacements of the live rows, `alpha` 1.2 keeps recall better than `alpha` 1.0 | On `a10k`, `recallLowest` of `graph-a12` is at least 0.005 above that of `graph-a10`, with at most 1.25 times its `distancesPerSearch` | not run |
| S2 | Merge by insertion into the largest segment, `MergeMode.INSERT` ([#16](https://github.com/humemai/tiervec/issues/16), [#161](https://github.com/humemai/tiervec/issues/161)) | It keeps the recall of a rebuild and writes cost less graph work | On `a10k`, `recallLowest` of `tier-insert` is at most 0.005 below that of `tier`, with at most 0.7 times its `distancesPerRowWritten` | not run |
