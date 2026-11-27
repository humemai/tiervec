# Study 106: the recall drop at 1,000,000 vectors on real embeddings

This is an edited copy of the plan I wrote on 2026-11-23, before any run. I removed the sections on resources, host setup commands, and computed runtime estimates (sections 8, 9, and 11), and I replaced host names and local paths. The question, the data, the arms, the rules, and the section numbers are the originals. The results are in [`results/106/real/`](../../results/106/real) and [`docs/EVIDENCE.md`](../../docs/EVIDENCE.md).

The question is #106: recall@10 of the unfiltered search at the default `ef` 100 is 1.0000 at 100,000 vectors and 0.5830 at 1,000,000 (`tvm-h1m`, which is `results/106/synthetic/h1m-jdk25.txt`) on the clustered-Gaussian data of the harness. A build with `--efc 400 --m 32` gives 0.8385 and one rebuilt segment 0.5700 (`tvm-diag1m`, the `h1m-diag` files). Is the drop the data or the build parameters? The issue says a real embedding set at 1,000,000 vectors would answer it, and the README status table of the repository gets no 1M recall row until it exists.

## 1. Data

Primary set: DBpedia entities with OpenAI `text-embedding-3-large` embeddings, 1,000,000 rows, 1536 dimensions, 26 parquet shards (8.9 GB; columns `_id`, `title`, `text`, `text-embedding-3-large-1536-embedding`, read from the footers on 2026-11-23). Source: the Hugging Face dataset `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M` (card: licence MIT, made from the first 1M entries of `BeIR/dbpedia-entity`; the texts are DBpedia's, whose own licence is a share-alike one, not checked here: publish recall numbers only, never the vectors or texts).

- **Why this one.** It is the only real set at 1M or more that can be made 768-dimensional with a rule I have already used: the first 768 coordinates, renormalised (datastax/jvector#709, 2026-11-01). text-embedding-3 is trained so that a prefix of the vector is itself an embedding, so this is the shortening the API offers (`dimensions`). 768 is the dimension of the synthetic run, so a vector costs the same 3 KB, the memory and the build work are those of `tvm-h1m`, and the numbers are comparable. The native 1536-dimension set needs about 20 GB of heap (BENCHMARKS.md says `-Xmx24g`), which fits neither machine I had for this.
- **Cut.** The set has 1,000,000 rows: no cut. 1,000 rows are held out as queries (random, seed 42, by the repository's prep script), so the index has 999,000 vectors, "1M" in the tables. The 100,000-vector arm is a seeded random subset of those 999,000 (`prep_set.py subset`, numpy `default_rng(42).choice`), with the SAME 1,000 queries and exact answers rescanned. A prefix of the file would not be a random sample of an ordered corpus.
- **Queries.** 1,000 (the synthetic runs used 200). The standard error of a mean recall over 1,000 queries is at most 0.5 / sqrt(1000) = 0.016 whatever the distribution; the arms share the queries, so a difference between arms is paired and smaller.
- **Exact answers.** `prep_real_dataset.py` computes them in float64 (a full scan). Two independent checks: `prep_set.py check` recomputes 20 queries with numpy over the whole file, and the harness compares the file with its own scan for the one filtered cell (`TRUTH-CHECK`, fails the run under 98 percent).

Optional control (b1), costs little: DEEP (`deep-image-96-angular`, Deep1B descriptors, 96 dimensions, angular, 9,990,000 rows, ann-benchmarks), as shipped (float32, not normalised). Cut: rows 0 to 999,999 are the index, rows 1,000,000 to 1,000,999 the queries (no randomness), the dataset's own queries and neighbours are not used (its neighbours are for all 9.99M rows). 384 MB of output. Licence: none stated by ann-benchmarks or the Deep1B authors (research data): numbers only. Why: a 96-dimension real set tells whether the tier is sound at 1M on real data at all (a failure there would be a bug, not a data effect). It cannot answer the question for 768 dimensions and no verdict rests on it. SIFT (128 dimensions, Euclidean) is not used: the harness is cosine only.

## 2. Pin and build

- tiervec `70b64f2e342430cb0c6389a187c35328219aca2e` (main when the real loader of #108 landed). `git diff 70d5dee6c3 70b64f2 -- src/main` is empty: the library bytes are those of `tvm-h1m`, plus `--data`. Main at the time of writing (`9128fa7`, #105) changes `VKern` (lane-sum order); it is NOT the pin, because an arm must differ from `tvm-h1m` only in the data. A rerun of a1 at main is a separate 30 minute question.
- Harness classes compiled by the JDK 21.0.12 `javac`, run on JDK 25.0.5 (the pair of `tvm-h1m`), `-Xmx13g`, `--build-threads 3` (what `tvm-h1m` ran with on 4 cores; the benchmark host has 6, and the default would give more threads), the Vector API module on.
- Recall of a fixed built graph and fixed queries is a function. The BUILD is not bit-deterministic (seals run on background threads at timing-dependent points, 3 build threads): `tvm-diag1m` d1 repeated `tvm-h1m` to 0.005 at ef=100 (0.5780 against 0.5830), the only pair there is.

## 3. Arms (one build each, `run.sh`)

All with `--queries 1000 --reps 3 --ef-sweep 20,50,100,200,400,800,1600,3200 --shares 0.01 --scan-below 0 --build-threads 3 --conc-seconds 0` (the cells of `tvm-diag1m`; the one filtered cell makes the harness scan and compare the answer file; no concurrent phase, which `--data` refuses).

| arm | data | build | answers |
|---|---|---|---|
| a0 | real, 100,000 vectors | M 16, efConstruction 200, adds and merges (the `tvm-h1m` build) | does the 1M drop need 1M? the 100k reference |
| a1 | real, 999,000 vectors | the same | does the drop repeat on real data? |
| a2 | real, 999,000 vectors | `--efc 400 --m 32` | does a stronger graph recover it on real data? |
| a3 | real, 999,000 vectors | `--load rebuild` (one segment) | CONDITIONAL, section 7: the merge and the eight segments |
| b1 | DEEP 96-d, 1,000,000 vectors | the `tvm-h1m` build | OPTIONAL control, no verdict rests on it |

Order: a0, a1, a2 (`run.sh main`), then a3 or b1 only as section 7 says.

## 4. Repetitions

What a repetition buys here: recall has no timing noise, so a second run of the same build with the same queries differs only by the non-deterministic build (0.005 in the one pair I have). The decision thresholds (0.03 and 0.10, section 5) are 6 and 20 times that pair, so one build per arm decides a clear case; a repetition is held for the unclear band and declared up front: ONE second build of a1 (`RUN_TAG=-r2`), nothing else is repeated. Latency (p50 of 3,000 timed searches per cell) is DIRECTIONAL: the lane is untimed, other jobs can run beside it, the cores are efficiency cores. This is not a timed study: no A/A arm, no latency claim beyond "about", and the `.env` and `.samples` files beside each result record the load, the temperature, and the throttle counters.

## 5. What is read and the rule (written before any run)

By `read_results.py`, from the files, no typed numbers: gates first (a file that fails one is a finding, no verdict), then recall@10 and p50 by ef per arm (the table of #106, with the synthetic runs of 2026-11-12 beside it), the build figures, and the smallest ef with recall at least 0.95. The deciding figures are recall@10 at ef=100 (the library default), unfiltered:

- D = recall(a0) - recall(a1), the drop with size on real data (synthetic: 1.0000 - 0.5780 = 0.4220).
- P = recall(a2) - recall(a1), what M 32 and efConstruction 400 buy at 1M on real data (synthetic: 0.8385 - 0.5780 = 0.2605).

Thresholds are those of the 2026-11-12 follow-up: 0.03 repeats or does not move (about two standard errors of one mean in the worst case, section 1), 0.10 moves.

| result | reading |
|---|---|
| D at most 0.03 | DATA. On real embeddings the default build holds at 1M; the 0.58 is a property of the synthetic distribution. The README status table can carry a real-data 1M row. No defaults issue |
| D above 0.10 and P above 0.10 | PARAMETERS. The drop is not specific to the synthetic data; M 16 and efConstruction 200 are small for 1M at 768 dimensions. The data sets the size (compare D and P with the synthetic ones). The defaults issue is warranted, with both tables |
| D above 0.10 and P at most 0.03 | OTHER. Neither the data nor these two parameters: run a3 and read the segment policy (#4) |
| anything else | UNCLEAR. One declared rerun of a1 (section 4); still in the band means "small effect", reported as that |

The b1 arm is read only against its own floor: recall at ef=100 below 0.90 on DEEP at 1M would be reported as a finding about the tier or the harness before anything else.

## 6. Validity gates (a file that fails one reports nothing)

`BENCH-SUMMARY status=OK failures=0` and `BENCH-END`; `COMMIT` sha equal to the pin with 0 modified files; `DATASET` count and dim as expected (100,000 or 999,000 vectors of 768; 1,000,000 of 96); zero `badAnswers`; `TRUTH-CHECK` overlap of at least 0.98; `cpusAllowedList` inside the allowed cores; JDK 25.x. `run.sh` refuses to start on a core outside `ALLOWED_CORES` (it reads `Cpus_allowed_list` back and never pins itself), refuses an existing result file, and refuses a kit that differs from `SHA256SUMS`.

## 7. Stopping rule, the conditional arm, rule changes

- a0, a1, a2 are the study. a3 runs only when the reading is OTHER or UNCLEAR. b1 runs when the lane is free after a2, once. The a1 rerun is the only repetition, and only in the UNCLEAR band. Nothing else is added after a result is seen; a different rule is a new study with a new folder.
- An arm that fails a gate is reported as a finding (bug or harness fault, to be told apart before anything is filed), and rerun only after the cause is named.

## 10. May and may not claim

May: recall@10 of the tier at these sizes on this real set (768-dimension prefix of DBpedia OpenAI embeddings, cosine), one build per arm, directional latency on efficiency cores, the one rerun if declared. May not: anything about other real sets, the native 1536 dimensions, other hardware, a speed, or "the defaults are right" beyond this set; no README row of the repository comes from this study before I have read it.
