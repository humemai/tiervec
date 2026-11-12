# Benchmarks

The harness is `src/bench/java/com/humemai/vec/Bench.java`. It builds a `TieredIndex` from seeded data, compares every answer with an exact scan that it computes itself, and writes one result file that names the machine, the JDK and the commit. It measures one question set: recall and latency of the tier at a size, unfiltered and filtered, and what happens when writers and searchers run at once. [Issue #12](https://github.com/humemai/tiervec/issues/12) (the range of allowed shares where an exact scan beats the graph walk) is the first user.

Numbers from the harness are directional unless a study states its claim, its arms, its number of runs and its bound before it runs (see "How to read measurements" in [ROADMAP.md](ROADMAP.md#how-to-read-measurements)). One run is one sample. A result file says what the run did on one machine, not what the library does on every machine.

## Run it

Maven does not compile the harness, as with `src/smoke`. Compile it with the library sources, and run it with the Vector API module (the module is optional for the library, but a result without it is a different measurement, and the file records which one ran):

```
mkdir -p target/bench/classes
find src/main/java src/bench/java -name '*.java' > target/bench/sources.txt
javac -d target/bench/classes --add-modules jdk.incubator.vector @target/bench/sources.txt
java -Xmx14g --add-modules jdk.incubator.vector -cp target/bench/classes com.humemai.vec.Bench \
  --host-label my-laptop --out results/$(date +%F)-my-laptop.txt
```

With no options this is the 1,000,000-vector run: dimension 768, cosine, `k` = 10, `ef` = 100, 200 queries with 5 timed passes, allowed shares 0.5, 0.1, 0.01 and 0.001 each with `filterScanBelow` off and on, then 10 minutes of concurrent writers and searchers. Every option is a `--key value` pair, and the list with the defaults is the method `Bench.defaults()`. The ones that matter most:

| Option | Default | Meaning |
|---|---|---|
| `--n`, `--dim`, `--k`, `--ef` | 1000000, 768, 10, 100 | Vectors, dimension, answers per query, beam width of a segment search |
| `--seed` | 42 | Seed of the data, the queries and the filter. The same seed gives the same vectors on every JVM (the `DATA` line has a checksum) |
| `--ef-sweep` | `20,50,200,400` | Extra unfiltered cells at these beam widths, a recall curve like the one of `GraphTest bench`. The recall floors do not apply to them |
| `--queries`, `--reps` | 200, 5 | Distinct queries, and timed passes over them per cell (after one warm-up pass that also measures recall) |
| `--shares` | `0.5,0.1,0.01,0.001` | Allowed shares of the filtered cells |
| `--scan-below` | `0,1` | `filterScanBelow` values of the filtered cells. 0 is off (the walk, with its fallback to a scan). 1 scans first at every share |
| `--tail-limit`, `--small-limit`, `--max-segments`, `--merge-fan`, `--build-threads` | the library defaults | The tier configuration. `--build-threads 0` keeps the default of `TieredIndex.Config` |
| `--load` | `add` | `add` calls `TieredIndex.add` n times and waits until no seal or merge is due. `rebuild` makes one `rebuildFrom` |
| `--settle` | `none` | `merge` calls `mergeNow()` after the build, so the searches run on one segment |
| `--conc-seconds` | 600 | Length of the concurrent phase. 0 skips it |
| `--writers`, `--searchers` | 2, 2 | Threads of the concurrent phase. There is one more thread that deletes |
| `--write-rate`, `--delete-rate` | 500, 100 | Adds and deletes per second over all threads. 0 for adds is unthrottled |
| `--min-recall`, `--min-scan-recall` | 0, 0 | Recall floors. With 0 the harness reports and does not fail on recall. The CI smoke sets both |
| `--host-label` | the host name | The name of the machine in the file. The file is meant to be committed to a public repository, so give a label that says what the machine is and not who owns it |
| `--out` | none | The result file. The same lines go to standard output |

The exit code is 0 when every check passed, 1 when a check failed (see "Checks" below), and 2 for a wrong option.

Memory, as an estimate that no run has confirmed yet: the harness holds the vectors (n times dimension times 4 bytes, 3.1 GB at the default), the tier holds its own copy of every vector, and a merge builds a new segment while it still holds the old ones. Allow about 10 GB of heap at 1,000,000 vectors of dimension 768 and give the JVM `-Xmx14g`. The `MEMORY` lines of a result file give what the run used.

## The data

- **Vectors.** A Gaussian mixture on the unit sphere: `--clusters` (100) unit centroids drawn from the seed, and the vector of an id is `normalize(centroid + sigma * noise)` with `--sigma` 0.4 (the generator `GraphT.Gen` of the tests, which `GraphTest bench` also uses). The cluster of an id and the noise are a function of the seed and the id only, and the Gaussian uses `StrictMath`, so the data does not depend on the machine.
- **Queries.** `--queries` vectors of the same distribution, with ids that are not in the index (from 2^50 up). They are not inserted.
- **Filter.** An id is allowed when a hash of the id and the seed, as a number in [0, 1), is below the share. The allowed sets of the shares are nested (the set of 0.01 is inside the set of 0.1), and the filter does not depend on the vector. A filter that is correlated with the vectors, such as one tenant per cluster, behaves differently and is not measured here.
- **Metric.** Cosine. The vectors have unit length, so the exact scan uses the dot product.

## The exact scan and recall

The harness computes the exact top `k` of every query with its own scalar dot product, in one pass over the vectors on all available cores. It does not use a kernel of the library, so a wrong kernel cannot hide behind itself. It does that for the unfiltered query and for every share. Recall@k of a query is the share of the exact ids that the answer contains (the truth is the allowed set when the query has a filter). The `SEARCH` line has the mean over the queries and the minimum. A tie at the boundary of the top `k` can swap one id, so recall of exactly 1 is not expected from an exact path: the score of an answer is compared with the exact score instead (`maxScoreError`).

The first pass over the queries in a cell is the warm-up and measures recall. The next `--reps` passes are timed, one search at a time on one thread, and give p50, p99, p99.9, mean, maximum and operations per second. A cell with `scanFirst=true` runs the exact scan of the allowed ids in every segment (the plan of [DESIGN.md](DESIGN.md#graph-index)).

## The result file

One run writes one file, `results/<date>-<host-label>.txt`, in lines of `key=value` pairs that start with a tag. It is plain text so that `grep` reads it. Nothing is written twice: a line is final when it is printed, and a file that does not end with `BENCH-END` is a run that did not finish.

| Tag | What it holds |
|---|---|
| `BENCH-RESULT`, `RUN` | The format number, the start time and the host label |
| `COMMIT` | `git rev-parse HEAD` of the repository at run time, and the number of modified tracked files (0 for a clean tree) |
| `SEED` | The seed and the data model |
| `CPU`, `CPUFLAGS`, `CPUFREQ`, `RAM` | The model of the first allowed core and its flags (a hybrid CPU has cores with different flags, so a run pinned to some cores names those), the cores the process may use, the frequency governor, the memory |
| `OS`, `JDK`, `JVM`, `VECTORAPI` | The operating system, the JDK version and vendor, the JVM arguments, the garbage collectors, and whether the Vector API module and the vector kernels were in use |
| `PARAMS`, `CONFIG` | Every option, and the tier configuration |
| `DATA`, `TRUTH` | Time to generate the data and to compute the exact answers, a checksum of the first vectors, and the smallest exact answer of each group |
| `BUILD`, `BUILD-STATS`, `BUILD-T` | Seconds to add and until the tier was settled, inserts per second, the tier statistics (seals, merges, segment sizes, write amplification, back pressure), and a progress line each minute |
| `MEMORY` | Heap in use and resident memory after a garbage collection, and the peak resident memory |
| `SEARCH` | One cell: kind, share, `filterScanBelow`, recall, latency percentiles, operations per second, malformed answers, garbage collections during the cell |
| `CONC`, `CONC-T`, `CONC-WINDOW` | The concurrent phase: adds, deletes and searches per second, search latency overall and per window, every error, a progress line per window |
| `CONC-SETTLE`, `CONC-CHECK`, `CONC-RECALL` | The end state: the live count against the expected count, `lastError`, and recall against an exact scan of the live set |
| `BENCH-SUMMARY`, `BENCH-FAILURE`, `BENCH-END` | `OK` or `FAIL`, the list of failed checks, and the end marker |

## Checks

The run fails (exit code 1) when:

- an exception reaches the harness or a thread of the concurrent phase;
- the live count after the build or after the concurrent phase is not the number of vectors that were added and not deleted (a lost or a phantom vector), or `lastError()` is set;
- an answer has more than `k` hits, scores out of order, an id twice, an id outside the data, or an id that the filter does not allow;
- a search that began after a delete had returned still finds that id (the deleter deletes the ids 0, 1, 2, ... in order, so the searchers know which deletes were done);
- a recall floor was set (`--min-recall`, `--min-scan-recall`) and a cell is below it.

## Rules for a run that is meant to be read

1. Pin the process to cores of one kind and run nothing else beside it. Write down the cores: the file records `cpusAllowedList`. A laptop on battery, with a turbo that varies, or with its low-power cores, is a directional measurement.
2. Use a JDK that CI tests (Temurin 21 or 25), name it in the file (`JDK`), and keep the Vector API module on, unless the run is the scalar one on purpose (`-Dhumemai.vec.scalar=true`).
3. A claim needs the runs that [ROADMAP.md](ROADMAP.md#how-to-read-measurements) asks for, each in a fresh JVM, and a bound that was written down before the first run. One run is a description.
4. Commit the file with its commit SHA inside, in `results/`, and say in the table row of the README what it covers.
5. Do not compare files from different machines or different JDK builds as if only the library changed.

## The CI smoke

The job `bench smoke (JDK 21)` and `(JDK 25)` compiles the harness with the library and runs it at n = 10,000 and dimension 768, with 100 queries, one extra beam width (32), shares 0.1 and 0.01, and 15 seconds of the concurrent phase. It is not a required check, and it runs beside the fast tests. It proves that the harness runs, that its answers agree with its own exact scan, and that the checks above hold on a small tier. It prints timings and has no speed bar: the runner is shared, so a timing is only a ratio inside one run.

The recall floors of the smoke are 0.90 for the unfiltered and the walk cells and 0.99 for the cells that scan first. In the first runs (CI run 43133522549, JDK 21 and 25) recall was 1.0000 unfiltered, 0.997 to 1.0000 for the filtered walk and 1.0000 for every scan-first cell, and a run took 21 to 23 seconds (the job about 30 seconds). The floors leave room for a different structure of the segments from one run to the next, and they are far above what a broken index gives: a tier that returned only its tail would find at most the 20 percent of the data that the tail holds, and the scan-first path is exact apart from a swapped tie at the boundary of the top `k`. A control step of the job runs the harness with a floor that cannot be met (recall of at least 1.5) and requires exit code 1 and a `BENCH-FAILURE` line, so the failure path is not a path that never runs.
