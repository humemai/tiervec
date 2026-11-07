# Host contract

This page states what a database (the host) must provide to embed tiervec, and what tiervec guarantees in return. It is written for an engine developer who reviews the contract before writing an adapter. The word provider appears in the file name: tiervec is the provider of a vector index, and the host is the database that calls it.

The library is experimental and is not wired into any database yet. Most lines below describe where the library is going, not where it is. Every line carries a mark, so a reader can tell a promise from a plan. Nothing here is a measurement: measurements will be published with the harness.

Two companion pages go deeper where a topic is large: [ID-EPOCH-AND-REPLAY.md](ID-EPOCH-AND-REPLAY.md) for ids, epochs, and recovery after a crash, and [DESIGN.md](DESIGN.md) for what the code does today.

## Purpose and non-goals

A host that embeds an index has to know what is atomic, who sees a write and when, which calls may block, which threads the library creates, and what happens after a crash. This page is the one place where those answers live, so a reviewer does not have to read the code.

The contract is written in host terms and names no particular database. The host-side interfaces use only `long`, `float[]`, `Map<String, String>`, `java.util.concurrent.Executor`, and `java.util.function` types, so the library needs no dependency on any database.

Non-goals:

- Transactions. The library has no rollback and no notion of uncommitted rows. They stay on the host side.
- Record storage. The host's records are the source of truth. Everything the library keeps is derived and can be rebuilt from them.
- A query language, a server, or a network. The host parses its own queries and hands the library vectors and allowed-id sets.
- Ordering across several indexes.

## Reading the marks

- `exists today`: built on the main branch at the time of writing, and checked by a test that the status table at the end names. The class and method are named.
- `partly`: a form of the guarantee exists, or it is built and no dedicated test checks it. The line says what is missing.
- `planned`: not built. The issue number is the tracking issue, and the acceptance test in that issue is the bar.

Names of planned interfaces and methods are descriptive, not final. The Java package is `com.humemai.vec` today, and a rename before the first release is still open (see [STATUS.md](STATUS.md)).

## Summary

What the host must provide:

1. Ids: an opaque `long` per vector version, minted by the host, unique inside an epoch, and strictly increasing for adds inside an epoch. The epoch changes whenever the host renumbers.
2. Apply order: one batch per committed transaction per index, applied only when the transaction's changes stand, serialized per index.
3. Records: a way to read every live record, to ask whether an id is live, and to learn the highest id the host itself recovered after a crash.
4. Resources: an executor, a build permit, and a memory budget, so the library creates no thread of its own, and a capacity check before each batch.
5. Transactions: the overlay of uncommitted rows, the mapping from ids to the host's records, and the translation of the host's filters into an allowed-id predicate.

What tiervec guarantees:

6. A batch is visible as a unit. After `apply` returns, a search that starts later sees the adds and not the deletes, and applying the same batch again changes nothing visible.
7. A search returns at most `k` hits, best first, with no duplicate id and no id deleted before the search started. The allowed-id predicate is applied inside the search. The query path takes no lock and creates no thread.
8. A search stops at a deadline or a cancel flag that the host supplies.
9. After a reopen, the persisted state is either reconciled with the host's records or discarded in favor of a rebuild. It never returns a vector for an id that now means a different vector.
10. Errors are unchecked and classified, and a contract version and a capability set tell the host what exists.

## Roles and host interfaces

The library defines three small interfaces for itself. A separate adapter, outside the library core, implements them on top of the host's own classes. No host class appears in the library's API.

| Interface | What the host supplies | Status |
|---|---|---|
| Environment | An executor for every seal, merge, and build task. A build permit: ask with a kind and an estimate of peak bytes, receive a permit or a refusal. A memory budget in bytes, read at each decision. A metric sink with counters and gauges. | `planned` ([#13]) |
| Record source | The host's epoch. Every live record. Every live record with an id above a given id. A liveness test for one id. The highest id the host recovered after its own crash. | `planned` ([#7]) |
| Search context | A deadline and a cancel flag, polled by the library during a search. | `planned` ([#11]) |

Today the library has none of the three. It runs one worker thread and, during a seal or a merge, a few build threads (`Config.buildThreads`), and it reads its records only through `add` and `rebuildFrom`.

## Operations

Each row names who does what. "Host" is the adapter on the host side, "library" is tiervec.

| Operation | Host does | Library guarantees | Status |
|---|---|---|---|
| `add(id, vector)` and `delete(id)`, single calls | Calls them with the rules of the identity section. | Each call is atomic and visible to the next search in any thread. `add` of a held id replaces its vector. `delete` returns `true` when the id existed. | `exists today`: `TieredIndex.add`, `TieredIndex.delete`, `VecIndex` |
| `apply(batch)` | Calls it with an ordered list of adds and deletes, once per committed transaction per index, serialized per index, with add ids strictly increasing inside the epoch. | The whole batch becomes visible together: adds are applied before deletes, under one hold of the write lock, and a search sees all of the batch or none of it. Re-applying a batch changes nothing visible. A search that overlaps a batch waits for it or runs again, so a batch can delay a search by its own length. | `exists today`: `TieredIndex.apply`. Tests: `ApplyBatchTest` (visibility, a search held across a batch, re-apply), `ApplyStressTest` (slow: updater and searcher threads) |
| `awaitCapacity(deadline)` | Calls it before `apply`, outside any lock of its own. | Waits for room in the tail or returns `false`. `apply` itself never sleeps on the library's maintenance. The behavior at the hard tail limit is defined and tested: `apply` does not look at the limit, and one batch may take the tail above it. | `exists today`: `TieredIndex.awaitCapacity`, with the deadline on the `System.nanoTime()` clock. Test: `ApplyBatchTest.behaviorAtTheHardTailLimit`. The single call `TieredIndex.add` keeps its own wait of up to 10 s at the hard limit |
| `search(query, k, allowed, context)` | Passes the query, `k`, an allowed-id predicate (optionally with a size hint), and a search context. | At most `k` hits, best first, no duplicate id, no deleted id, fewer than `k` only when fewer qualify. A search stops when the context says so. | `exists today` without the context: `TieredIndex.search`, three overloads. Context: `planned` ([#11]). Size hint: `planned` ([#12]) |
| `open(config, fingerprint, environment, record source, directory)` | Passes the index definition (dimension, metric), the configuration, its fingerprint, the interfaces above, and a directory that only this node uses (or none, for memory only). | Returns ready or cold. A mismatch of fingerprint, dimension, metric, or epoch discards persisted state and returns cold, never a wrong answer. | `planned` ([#9], [#7]) |
| `awaitReady(deadline)` | Calls it before the first search of a cold index, and from its own warm-up hook. | Builds from the record source on the host executor under a host permit, never on the caller's query thread unless the caller asks for that. | `planned` ([#9], [#13]) |
| `rebuildFrom(records, epoch, snapshot position)` | Calls it for a first build, a rebuild, the fallback after corruption, and after a renumbering. | Replaces all derived state without blocking `apply` or `search`. Adds and deletes that arrive during the build are kept and applied after the swap. Streams the records instead of copying them first. | `partly`: `TieredIndex.rebuildFrom(Iterable<IdVector>)` is complete and correct, but holds the write lock for the whole build, copies all records into a map first, and has no epoch and no snapshot position. Rest: `planned` ([#7]) |
| `flush()` | Calls it where the host flushes its own indexes. | Writes the manifest and any dirty state, returns when durable, idempotent. | `planned` ([#9]) |
| `close()` | Calls it when the host closes the index. | Stops accepting work, waits for or cancels the tasks it submitted, releases files, idempotent, never creates a thread. | `partly`: `TieredIndex.close()` stops the worker thread and a second call is harmless. No defined behavior for searches in flight. Rest: `planned` ([#13], [#18]) |
| `drop()` and `clear()` | Calls `drop` when the index is deleted and `clear` when its data is truncated. | Removes every file it wrote and all in-memory state, idempotent. | `planned` ([#9]) |
| `verify()` | Calls it from the host's own integrity check. | Returns a list of problems and never throws for a damaged file. | `planned` ([#9], [#17]) |
| `onRenumber(newEpoch)` | Calls it at the point where it reissues ids. | Drops derived state and rebuilds in the new id space. | `planned` ([#7]) |
| `metrics` | Reads it into its own statistics. | A map of counters and gauges, plus an optional push sink. | `partly`: `TieredIndex.stats()` returns a `Stats` record. Map and sink: `planned` ([#18]) |
| `contractVersion()`, `capabilities()`, `health()` | Checks them at creation and when index metadata changes. | A major and a minor number, a capability set, and a health state. | `planned` ([#18]) |
| Suspend and resume of background maintenance | Calls it around a bulk load. | No seal or merge starts while suspended. `apply` still works. | `planned`, no issue filed yet |

## Identity and the epoch

The library keys every vector by an opaque `long` id that the host supplies. The id is not a record id of the host: a record id can be reused after a delete, can be wider than 64 bits, and can change when the host moves a record. An id inside an epoch cannot be reused.

- The host mints ids. They are unique inside an epoch, and the ids of adds strictly increase inside an epoch. Gaps are allowed. `exists today`: ids are plain `long` values in `VecIndex.add`, `Hit`, and `TieredIndex.IdVector`. The ordering rule is `planned` ([#7]).
- An update is a new id plus a delete of the old id, sent in one batch. The library needs nothing special for it: two ids, and the delete hides the old copy. `exists today` as one batch (`TieredIndex.apply`) and as two calls.
- The epoch names one numbering of ids. The host passes it to `open` and `rebuildFrom`. The manifest stores it. A different epoch means persisted state is discarded and the index starts cold. `planned` ([#7], [#9]).
- A host that reuses ids anyway (a record id used directly as the key) is served in memory by the version floor: an id added after a delete is visible, and the old copies stay hidden. `partly`: `TieredIndex.add`, `TieredIndex.delete`, and the floor described in [DESIGN.md](DESIGN.md) are built, and the tests cover replace, not delete followed by add of the same id. Tests for reuse across seal and merge are `planned` ([#5]). Persisting such a tier is not covered by this contract.

The id models, their consequences, and the open questions are in [ID-EPOCH-AND-REPLAY.md](ID-EPOCH-AND-REPLAY.md).

## Mutation, replay, and ordering

- Idempotence. `add` of a held id replaces it with the same vector, and `delete` of an absent id returns `false` and changes nothing. A host may therefore replay a batch after a crash, and replaying one that was already applied changes nothing visible. `exists today` for single calls: `TieredIndex.add`, `TieredIndex.delete`. `exists today` for whole batches: `TieredIndex.apply`, tested by `ApplyBatchTest.applyingTheSameBatchAgainChangesNothingVisible`.
- Order. Inside a batch, adds are applied before deletes. Across batches, the host serializes calls per index. The library needs no remove-before-add rule, because ids are unique. `exists today`: `TieredIndex.apply`, tested by `ApplyBatchTest.theWholeBatchIsVisibleAfterApplyReturns` (an id that one batch adds and deletes ends deleted).
- Apply at commit, or apply after validation. The recommended host behavior is to stage a transaction's changes and call `apply` only when the changes stand. The library then never sees an id of a rolled-back transaction and needs no retract operation. The reason is that a sealed segment is immutable and can be made durable on its own schedule: a row applied too early and then rolled back can be sealed before the rollback runs. The floor hides it from results, but the segment still holds a row for an id the host has forgotten, and the host may hand that id to another record after a restart. The library has no transaction notion, so nothing in it contradicts deferred apply. `exists today` on the library side, as a property of the code with nothing to test. Staging is the adapter's job.
- A host that cannot defer can undo an add with `delete` and undo a delete by adding the vector again. The version floor makes that correct in memory. The library has no retract operation. The capability set will say which style a given library version supports. `planned` ([#18]).
- Replay after a reopen: the host replays every live record above the manifest's high-water id, and the library checks liveness of every id it holds. Specified in [ID-EPOCH-AND-REPLAY.md](ID-EPOCH-AND-REPLAY.md). `planned` ([#7], [#9]).

## Search

- Arguments: a query vector, `k`, an allowed-id predicate (or none), and a context. An optional effort knob sets the beam width of the segment search. A non-positive `k` returns an empty result. `exists today` without the context: `TieredIndex.search(float[], int, LongPredicate)`, `search(..., Freshness)`, `search(..., int ef)`.
- Result: at most `k` hits, best first, each `Hit(id, score)`, no duplicate ids. Fewer than `k` only when fewer live vectors qualify. `exists today`: `VecIndex.search`, `Hit`.
- Score: higher is better for every metric. Cosine and dot are returned as they are, L2 as the negative squared distance. A host whose engine uses a distance where lower is better converts, and checks the conversion with a parity test on the same data. `exists today`: `Hit`. A public scorer for a single pair of vectors, which a host needs to score its own uncommitted rows with the same arithmetic, is `planned`, no issue filed yet.
- Exactness flag: `SearchResult.exact` tells the host that no segment was searched and the tail scan was exact. `exists today`: `TieredIndex.search(..., Freshness)`.
- Allowed-id predicate. The host supplies a `LongPredicate`. It must be cheap, side-effect free, safe to call more than once for the same id, and it must not call back into the index. The library applies it inside the segment search and the tail scan, so the top `k` of the qualifying set is correct, and a graph walk with few qualifying ids falls back to an exact scan after a visit budget. `exists today`: the predicate parameter of `TieredIndex.search`. Correctness of filtered search is checked by unit tests. How it behaves across the range of selectivities has not been measured. An optional size hint and a live count, so that a small allowed set is scanned exactly, are `planned` ([#12]).
- Context. The library polls the context at a fixed, documented interval inside the tail scan and the segment search and between segments, and throws `SearchCancelledException` when it says stop. `planned` ([#11], [#18]).
- Threads and locks. A search runs on the calling thread, takes no lock, and never waits for a swap or a build. `partly`: built in `TieredIndex.search` and stated in its class documentation, but no dedicated test checks it.
- Freshness. By default a search sees the tails, so a write is visible at once. `Freshness.SEALED_ONLY` skips the tails and may lag. It never returns a deleted id, a stale vector, or an id that was never added. `exists today`: `TieredIndex.Freshness`.

## Visibility and publish

- The writing transaction sees its own uncommitted rows through the host's overlay. Any other session sees nothing before commit. After commit, the batch is visible from the moment `apply` returns. A host that calls `apply` before it tells its client the commit is done gives that client read-your-writes. The overlay is the adapter's job, see the last section.
- The segment list, the active tail, and the frozen tail live in one immutable view. A seal or a merge publishes a new view with one volatile write, and a search reads the view once. `exists today`: the view inside `TieredIndex`.
- An in-flight search finishes on the view it read, and retired segments stay valid until the last reader leaves. In memory, garbage collection does that. `partly`: there is no dedicated test of a long search across a swap. A memory-mapped reader would need a reference count before it unmaps, and none is planned yet.
- Seal and merge never block a search, and block `add`, `delete`, and `apply` only for a short critical section. `partly`: true for seal and merge, but `rebuildFrom` holds the write lock for the whole build ([#7]).
- Close with searches in flight: later calls throw `ClosedException`, and an in-flight search finishes or throws it, never returning a wrong id. `planned` ([#18]).

## Lifecycle

- Open: the host opens after its own recovery has finished, so the records and ids it exposes are final. The result is ready or cold.
- Readiness: a cold index is built from the record source, on the host executor, under a host permit.
- Rebuild from records: always available, and always correct, because the vectors in the host's records are the truth. It is the answer to every integrity failure.
- Flush, close, drop, clear, and verify: as in the operations table.
- Files: a host may keep the library's files outside its own storage. The library never registers a file with the host's storage manager, writes only inside the directory it is given, and treats a restored or replicated copy that has no files as a cold index. `planned` ([#9]).
- Persistence: the format, as the first cut writes it, is in [SEGMENT-FORMAT.md](SEGMENT-FORMAT.md). A settled tier is written by `TieredIndex.checkpoint` and read by `TieredIndex.open`. The tests for damaged files and crashes at scale are [#17].

## Recovery

- After a restart the host replays what the persisted state lacks, and the library reconciles what the host deleted or never made durable. The steps, the high-water id, the liveness check, and the durable-prefix rule are in [ID-EPOCH-AND-REPLAY.md](ID-EPOCH-AND-REPLAY.md). `planned` ([#7], [#9]).
- Any integrity failure at open throws, and the host calls `rebuildFrom`. A second failure is a host error. `partly`: `TieredIndex.open` throws `SegmentFormat.CorruptException`, a checked `IOException` (and `UnsupportedException`, `MismatchException`, `NoSuchFileException`), and an overload that takes a rebuild source does the fallback. The unchecked `CorruptStateException` of the error classes is [#18].
- The claim: a process kill leaves the last published generation or the one before it. No claim is made about power loss until a test says otherwise. `planned` ([#17]).

## Resources

- No thread of the library's own. Every seal, merge, and build task, including the parallel part of a build, goes to the host executor. `partly`: today `TieredIndex` starts one worker thread and, per build, a few threads (`Config.buildThreads`). `planned` ([#13]).
- Build permit. Before each seal, merge, and rebuild the library asks for a permit with an estimate of peak bytes, computed from the sizes of the segments involved. On refusal it defers and retries with back-off, and never builds without a permit. `planned` ([#13]).
- Memory budget. The library keeps its resident bytes below the budget by deferring merges and shrinking its tail limit. `partly`: `TieredIndex.bytesHeld()` reports resident bytes, and nothing reads a budget yet. `planned` ([#13]).
- Throttle. The insert rate of a merge build is capped (`Config.mergeRatePerSec`). It is a politeness limit for the threads that serve queries, not a substitute for a permit. `partly`: built, and no dedicated test checks the cap.
- Back pressure is a pre-check (`awaitCapacity`), and the tail is bounded. `partly`: `TieredIndex.awaitCapacity` is built and tested. The bound is soft: `apply` does not look at it, so one batch can take the tail above the hard limit, and `TieredIndex.add` adds after its own wait of up to 10 s.
- No search-time thread creation, and no allocation proportional to the collection size on the query path. `partly`: built in `TieredIndex.search`, and no dedicated test checks it.
- The graph kernels need the `jdk.incubator.vector` module at run time. A scalar twin is `planned` ([#14]), and a native-image build is `planned` ([#15]).

## Configuration

- The library reads its configuration from a `Map<String, String>` and a fingerprint. The fingerprint is a canonical string and a hash over the data-affecting settings only: dimension, metric, id width, graph parameters, segment quantization, and layer layout. Search-time knobs (the beam width, the split-beam switch) and the settings of the tail are not part of it, because the tail is not persisted. `planned` ([#18]).
- The host stores the fingerprint in its index metadata and passes it back at open. A mismatch discards persisted state.
- Unknown keys are rejected at creation with the key named, and ignored with a warning at load, so a downgrade still opens.
- Search-time knobs may change at run time. Data-affecting keys may not.
- `partly`: the typed `TieredIndex.Config` record with validation, `with...` copies, and `Config.defaults()` is built, and no dedicated test checks the validation. The knobs are in [DESIGN.md](DESIGN.md).

## Metrics

- A pull form: a `Map<String, Long>` of counters and gauges. A push form: a sink with `counter(name)` and `gauge(name, supplier)`. Names are prefixed `tiervec.`. No metrics library is a dependency. `planned` ([#18]).
- The quantities are the ones `TieredIndex.Stats` holds today: seals, merges, failures, back-pressure waits, tail size, segment count, bytes held, floor entries, and write amplification (a gauge in thousandths). `partly`: `TieredIndex.stats()`.

## Errors

- Unchecked only. The base is a provider exception. Classes: `CorruptStateException` (the host falls back to `rebuildFrom`), `SearchCancelledException`, `ClosedException`, and `BuildRefusedException`. Programming errors stay `IllegalArgumentException` and `IllegalStateException`. `partly`: the two standard exceptions exist today (a dimension mismatch throws `IllegalArgumentException`). The classified family is `planned` ([#18]).
- A background failure never throws into `apply` or `search`. It is counted, kept as the last error, and retried. A health state reports ok, degraded (a task failing repeatedly), or failed (the worker is dead). `partly`: `TieredIndex.lastError()`, `TieredIndex.workerAlive()`, and the `failures` count in `Stats`. `planned` ([#18]).

## Versioning and capabilities

- The contract has a major and a minor number. The host refuses a library with a larger major. A larger minor is accepted, and capabilities it does not know are ignored. New operations are added as default methods. `planned` ([#18]).
- The segment file and the manifest have their own version numbers, under the same rule. `planned` ([#6], [#9]).
- Capabilities are a bit set: deferred apply, selective plan, persistence, and snapshot rebuild, with more as they appear. The host checks them at creation. `planned` ([#18]).
- What tiervec does not offer, so a host can refuse such an index definition at creation and not at query time: grouped (per-group top `k`) search, and approximate scores from compressed codes without an exact rescoring pass. Returned scores are exact fp32 scores. `exists today` as a fact about the code. The check through capabilities is `planned` ([#18]).

## What a host adapter typically does

The adapter is the host's code. It sits between the host's classes and the library, and it is where the host's own concepts end.

- Key mapping. The adapter keeps a table from library id to the host's record handle, and mints a new id for every new vector version. A hit costs one lookup in that table. An update sends an add of the new id and a delete of the old id in one batch.
- Overlay of uncommitted rows. The library has no transaction notion. For a search inside an open transaction, the adapter scores that transaction's pending rows exactly, with the same metric, merges them with the library's hits by score, and masks the ids the transaction has superseded by excluding them in the allowed-id predicate.
- Allowed-id bridging. The host's filter produces a set of record handles. The adapter wraps it as a `LongPredicate` over library ids: map the id to its record handle, then test membership. It passes a size hint when it knows one.
- Staging. The adapter holds a transaction's adds and deletes until the changes stand, then calls `apply` once.
- Recovery glue. The adapter implements the record source on top of the host's own record scan and id table, and persists the epoch beside its own data.
- Conversion. If the host reports distances where lower is better, the adapter converts scores, and a parity test on the same data checks the conversion.

## Status of every guarantee

Rows marked `exists today` or `partly` name the class and method, and the test that checks them where one exists. Tests are in `src/test/java/com/humemai/vec`. `TierTests` is the tier program, run one test at a time by `TierSuiteTest` in the fast job, and `TierConcurrentLongTest` runs the long threaded test in the slow job.

| Guarantee | Status | Class and method | Checked by | Issue |
|---|---|---|---|---|
| Opaque `long` ids, replace by adding a held id | `exists today` | `VecIndex.add`, `TieredIndex.add`, `Hit` | `TierTests.testTailOnly`, `TierTests.testDeleteReplace` (a replaced id is found by its new vector) | |
| `delete` returns `true` when the id existed and `false` otherwise, and a deleted id is never returned | `exists today` | `TieredIndex.delete` | `TierTests.testTailOnly`, `TierTests.testDeleteReplace`, `TierConcurrentLongTest` | |
| Read-your-writes for single calls | `exists today` | `TieredIndex.add`, `TieredIndex.search` | `TierTests.testSealStress` (the writer and another thread find a vector right after `add` returned), `TierConcurrentLongTest` | |
| Reused id visible, old copies hidden, in memory | `partly` | the version floor in `TieredIndex` | `TierTests.testDeleteReplace` covers replace, not delete followed by add of the same id | Reuse across seal and merge: [#5] |
| Search arguments, result order | `exists today` | `TieredIndex.search`, `VecIndex.search` | `TierTests.testTailOnly` (equals brute force), `TierTests.testDeleteReplace` (sorted, no deleted id) | |
| Score convention | `exists today` | `Hit` | `TierTests.testMetrics` (returned scores equal the exact score for each metric) | |
| Allowed-id predicate inside the search | `exists today` | `TieredIndex.search` | `TierTests.testTailOnly` (equals brute force), `TierTests.testSegmentsRecall` (filtered recall bound) | Selectivity study and size hint: [#12] |
| Atomic publish of the segment list | `exists today` | the view inside `TieredIndex` | `TierTests.testSwapInstantHooks` (no vector missing just before or just after a swap) | |
| No lock and no thread on the query path | `partly` | `TieredIndex.search` | no dedicated test: the code takes no lock, and `TierConcurrentLongTest` runs searches during seals and merges | |
| Rebuild from records | `partly` | `TieredIndex.rebuildFrom` | `TierTests.testRebuild` | Epoch, snapshot position, non-blocking: [#7] |
| Close stops the worker, second call harmless | `partly` | `TieredIndex.close` | `TierTests.testClose` | Searches in flight, tasks it submitted: [#13], [#18] |
| Statistics | `partly` | `TieredIndex.stats` | `TierTests.testMerge` and `TierTests.testSealStress` assert fields of it | Map and sink: [#18] |
| Background failure kept and counted | `partly` | `TieredIndex.lastError`, `Stats.failures` | no dedicated test: the tier tests assert that `failures` stays zero | Health state: [#18] |
| Merge throttle | `partly` | `Config.mergeRatePerSec` | no dedicated test | |
| Atomic batch, `awaitCapacity`, re-apply | `exists today` | `TieredIndex.apply`, `TieredIndex.awaitCapacity` | `ApplyBatchTest` (visibility, a search held across a batch, re-apply, the hard tail limit), `ApplyStressTest` (slow) | A batch can delay a search by its own length |
| Search context, deadline, cancellation | `planned` | | | [#11] |
| Size hint and live count for the allowed-id predicate | `planned` | | | [#12] |
| Host executor, build permit, memory budget, no own threads | `planned` | | | [#13] |
| Epoch, high-water id, replay, liveness check, durable prefix | `planned` | | | [#7] |
| Segment file and manifest specification | `partly` | [SEGMENT-FORMAT.md](SEGMENT-FORMAT.md) (the format as the first cut writes it) | `SegmentFormatTest.sectionBytesMatchTheInMemoryCounter` (size formulas against the in-memory counter) | Review of the format: [#6] |
| Persistence, open, flush, drop, clear, verify | `partly` | `TieredIndex.checkpoint`, `TieredIndex.open`, `SealedGraph.serialize`, `SealedGraph.load` | `TierReopenTest`, `ManifestCrashTest`, `SegmentFormatTest` | A settled tier only, no tail replay, no non-empty tombstones, no memory mapping. `flush`, `drop`, `clear`, `verify`, epoch and high-water id: [#7], [#9] follow-ups |
| Corruption and crash tests | `planned` | | | [#17] |
| Configuration map, fingerprint, capabilities, contract version, error classes, health | `planned` | | | [#18] |
| Scalar twin of the graph kernel | `planned` | | | [#14] |
| Native-image build and smoke test | `planned` | | | [#15] |
| Public scorer for one pair of vectors | `planned` | | | no issue filed yet |
| Suspend and resume of background maintenance | `planned` | | | no issue filed yet |

[#5]: https://github.com/humemai/tiervec/issues/5
[#6]: https://github.com/humemai/tiervec/issues/6
[#7]: https://github.com/humemai/tiervec/issues/7
[#9]: https://github.com/humemai/tiervec/issues/9
[#10]: https://github.com/humemai/tiervec/issues/10
[#11]: https://github.com/humemai/tiervec/issues/11
[#12]: https://github.com/humemai/tiervec/issues/12
[#13]: https://github.com/humemai/tiervec/issues/13
[#14]: https://github.com/humemai/tiervec/issues/14
[#15]: https://github.com/humemai/tiervec/issues/15
[#17]: https://github.com/humemai/tiervec/issues/17
[#18]: https://github.com/humemai/tiervec/issues/18
