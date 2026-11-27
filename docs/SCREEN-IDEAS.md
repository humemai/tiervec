# Screen ideas

Every idea for the index is listed here before its first run, with its source, what it is expected to do, and the bar it has to pass. The result is added after the run. An idea that fails stays in the list with its numbers, so that it is not tried twice.

The ideas are tested in two stages ([plan on #159](https://github.com/humemai/tiervec/issues/159#issuecomment-7222822816)): many quick rounds on the [screen](AGENT-MEMORY-BENCHMARK.md#the-screen), at small sizes, and then a confirmation at larger sizes only for the ideas that pass, under rules written before that run. A screen has no timing, and its sizes and data are small and generated, so no default changes on a screen result alone.

## The data of the screen

The [first run](https://github.com/humemai/tiervec/actions/runs/46041361120) used 128 dimensions with noise in all of them and queries near stored rows. It ran the exact and tier arms only, and their lowest recall was 0.999 or more at every size, so a difference between two designs had no room to show.

Harder data is chosen once, by this rule, written before any run on it. Only the baseline arms run while it is chosen (`exact`, `tier`, and `graph-a10`, the live graph at `alpha` 1.0), never an idea.

- The candidates are 768 dimensions with `--latent` 32, 64, 128, 256, and 768, in this order ([the two kinds of data](AGENT-MEMORY-BENCHMARK.md#the-runbook)).
- The choice is the first candidate at which, on `a10k` with seed 1 and the write mix, `recallLowest` of both `tier` and `graph-a10` is at most 0.97. If none is, the choice is 768.
- The chosen data, its difficulty (the `DATA` lines), and the runs of every candidate that was tried are recorded here.

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
