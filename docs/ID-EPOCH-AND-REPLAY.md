# Id epoch and replay

This page specifies how a host numbers the vectors it gives to tiervec, how it tells the library that the numbering changed, and what the host and the library do after a restart. It belongs to the [host contract](PROVIDER-CONTRACT.md) and answers the questions that page leaves open about ids and recovery.

Status: specification. Nothing on this page is built except the pieces named as `exists today`. The tracking issues are [#7] (epoch, snapshot rebuild, replay) and [#9] (the manifest that stores the epoch). Nothing here is a measurement.

## Why it exists

The library keys every vector by an opaque `long` id that the host supplies. A host can renumber its ids (for example when it compacts its own storage), can reuse them, and can replay writes after a crash. A persisted segment that does not know which numbering its ids belong to can, after a restart, return the wrong vector for an id, and nobody notices. Without a replay protocol, every host invents its own recovery and gets the edge cases differently wrong.

The library is meant to hold derived data only. The vectors in the host's records are the truth, so every rule below is built to keep that true: whatever the library persisted can be checked against the records, and thrown away when the check fails.

## Terms

- Id: the `long` that the host passes to `add` and `delete` and receives back in `Hit`.
- Epoch: a `long` that names one numbering of ids. The library compares epochs only for equality and never orders them.
- High-water id: the largest id held by any sealed segment of the persisted state. Every id in the tail is above it. This holds because a seal takes the whole frozen tail, and because add ids increase inside an epoch.
- Recovered maximum: the highest id the host itself found after its own recovery.
- Cold: an index with no usable persisted state. It is built from the host's records.

## Id models

Three models are possible. They differ in what the key is, and so in what the host and the library must do about reuse, ordering, and renumbering.

| | Stable host id plus epoch (recommended) | Host record handle as the key | Dense id that a compaction can reissue, with no epoch |
|---|---|---|---|
| What the key is | A `long` that the host mints per vector version, never reused inside an epoch, increasing for adds. | The host's own record handle, packed into a `long`. | The host's dense ordinal, `0` to `n - 1`, reissued in rank order when the host compacts. |
| Width | Fits a `long` by construction. | A handle can be wider than 64 bits. Packing needs a width limit and an overflow check, and a wider key changes every id array in the segment format. | Fits a `long`. |
| Reuse of a record handle | Impossible inside an epoch: a record that reuses its handle gets a new id. | Possible. A freed handle can come back for another vector, so a delete and a later add carry the same key. Correct only through the version floor. | Not an issue between compactions. |
| Update | An add of the new id plus a delete of the old id, in one batch. The floor degenerates to a set of dead ids. | An add of the same key replaces the old copy, and the old copy is hidden by a floor raise. | As the stable id, between compactions. |
| Delete on disk | `delete(id)`. Tombstones on disk are an optimization, because open checks liveness. | A floor raise per key. After a reopen the floor must be rebuilt from durable delete events in order, which needs a position in the host's log and an ordered iterator over it. | As the stable id. |
| Ordering rule for the library | None: ids are unique. | The host must send the delete before the add of the same key, and must supply a version per operation, because a counter inside the library reissues versions on replay. | None, between compactions. |
| Renumbering | Detected: the epoch changes, and persisted state is discarded. | A host that moves records changes keys, and nothing detects it. | Not detected. Persisted state silently refers to the old numbering, and id 7 may now be another vector. |
| What the library changes | An epoch parameter on open and rebuild, and an epoch and a high-water id in the manifest. The in-memory tier is unchanged. | A persisted floor, the key packing, and an optional section that carries the host's handle for each row, which can go stale. | Nothing, and that is the problem. |
| Residual risk | The epoch must survive a crash that falls between the host's renumbering and the library's notice. | Reuse across a reopen is closed only by replaying the host's log in order, which is host work. | Persisted state is unsafe. Only a memory-only index, rebuilt at every open, is safe. A content fingerprint (live count, next id at the last checkpoint, a hash of the live ids in order) detects a renumbering with high probability and does not prove it. |

The recommended model is the first. Reasons, all from the table: no reuse hazard inside an epoch, no width problem, no reconstruction of a floor from events the host may not store, a host mapping that usually exists already (the host finds a record from an id with a table it keeps), and no change to the in-memory tier.

`exists today`: the in-memory tier works with all three models, because the version floor hides old copies of a reused id (`TieredIndex.add`, `TieredIndex.delete`, [DESIGN.md](DESIGN.md)). Tests for reuse across seal and merge are `planned` ([#5]). The epoch, the high-water id, and persistence are `planned` ([#7], [#9]).

With persistence on, one more rule follows from the high-water id. A host does not add a held id again with a different vector. In memory, the tier would replace it. After a reopen, the replay rule would miss a replaced vector whose id is at or below the high-water id, and the old copy could come back. A new vector version gets a new id. Adding a held id with the same vector is the replay case, and it is allowed.

## The epoch rule

1. The host passes an epoch to `open` and to `rebuildFrom`.
2. The epoch changes whenever the host reissues or renumbers ids, including a compaction that renumbers, a restore from a backup that renumbers, and an import. It does not change when the host rebuilds an index and keeps its ids.
3. Inside an epoch, the ids of adds strictly increase in the order the host applies them. Gaps are allowed. An id is not reused inside an epoch, not even after a delete.
4. The manifest stores the epoch. At open, an epoch that differs from the host's gives a cold index. Persisted state is discarded as a whole, never partly reused.
5. When the host renumbers while the library is running, it calls `onRenumber(newEpoch)` at the point where it reissues ids. The library drops its derived state and rebuilds in the new id space.
6. The host persists the epoch together with its own data, in the same atomic step as the renumbering, so that a crash cannot leave new ids with the old epoch.

`planned` ([#7], [#9]).

## The replay protocol

The host replays what the persisted state lacks, and the library reconciles what the host deleted or never made durable. This is the sequence after a restart, with the host's side marked as host.

1. Host: finish the host's own recovery first, so that its records, its id table, and its recovered maximum are final. The library opens after that.
2. Host: call `open` with the index definition, the configuration and its fingerprint, the environment, the record source, and the directory. The record source offers the epoch, every live record, every live record above a given id, a liveness test for one id, and the recovered maximum.
3. Library: find the newest valid manifest. A damaged or truncated manifest falls back to the previous generation, and when none is usable the library throws `CorruptStateException`, so that the host calls `rebuildFrom`.
4. Library: compare the manifest's fingerprint, dimension, metric, and epoch with the host's. Any mismatch gives a cold index and ends the sequence here.
5. Library: apply the durable-prefix rule. Every held id above the recovered maximum is dropped (see below).
6. Library: for every id still held in a segment, ask the record source whether the id is live. A record that is not live marks the row dead. This is a check per held id, not a scan of the host's pages.
7. Library: ask the record source for every live record above the high-water id, in increasing id order, and append the vectors to the tail. Segments are not touched.
8. Library: report ready. The host resumes normal traffic: one `apply` per committed transaction.

If the persisted state is cold, the host calls `rebuildFrom` over its live records, with its epoch and a snapshot position, and the library reports ready when the build is swapped in.

Rebuild while the host keeps writing: the host passes a snapshot position, which is the highest id covered by the records it hands over. The library builds without blocking `apply` or `search`. Adds above the snapshot position that arrive during the build are kept and applied after the swap. Deletes that arrive during the build are kept as well, in arrival order, and applied after the swap, so that a delete of an id that the snapshot contains is not lost. None is lost and none is applied twice. `partly`: `TieredIndex.rebuildFrom(Iterable<IdVector>)` is correct and complete, and it holds the write lock for the whole build. The snapshot position, the epoch, and the catch-up are `planned` ([#7]).

## The idempotence rule

`add` of a held id replaces it with the same vector, and `delete` of an absent id returns `false` and changes nothing. Re-applying a batch therefore changes nothing visible, on one condition that the host guarantees: inside an epoch, an id always denotes one vector.

Because of this, the host may start a replay at or below the true position. A replay that starts too early is slower, never wrong. This is what lets the high-water id lag, and what makes a crash in the middle of a replay harmless.

`exists today` for single calls: `TieredIndex.add`, `TieredIndex.delete`. `exists today` for whole batches: `TieredIndex.apply`, tested by `ApplyBatchTest.applyingTheSameBatchAgainChangesNothingVisible`.

## The durable-prefix rule

A commit can count as final in the host's view and still miss the disk (for example under an asynchronous log flush). A segment that the library persisted earlier can then hold an id that the host forgot and hands to another record after the restart.

Ids increase in the order in which the host applies batches, so the ids the host lost are exactly the ids above its recovered maximum. At open, the library drops every held id above that maximum.

Two consequences follow from the way the library persists state. They are part of the rule.

- Dead is sticky. A row that an open or an `apply` has marked dead stays dead in every later generation. A liveness check may kill rows and never revives one. Otherwise, after the host mints a new record with a dropped id, a later open would find that id live again and bring back the old vector.
- The high-water id is lowered to the recovered maximum when rows are dropped. The host may mint the dropped ids again for new records, and those records must lie above the high-water id, so that a later replay finds them.
- The drop must be durable before the library reports ready, or the library discards the persisted state and goes cold. The library may always choose cold. The first implementation may do so whenever the rule drops a row.

`planned` ([#7], [#9]).

## What a crash leaves, at each point

A process kill is assumed. No claim is made about power loss until a test says otherwise ([#17]).

| The crash falls | What is durable | What the next open does |
|---|---|---|
| In the host, before its commit is durable | Nothing of this batch, in the host or in the library. | Nothing to do. The library never saw the batch, because the host applies only when changes stand. |
| After the host's commit is durable and before `apply` ran | The host's records. | Step 7 of the replay: the live records above the high-water id come back into the tail. |
| After `apply`, before any seal | The host's records. The tail was in memory only. | As above. |
| During a seal or merge, before the new manifest is durable | The previous generation, plus an orphan segment file. | The previous generation opens. The orphan is deleted. The records above its high-water id are replayed. |
| After a new manifest is durable, and the host lost the commit that its segment holds | A segment that holds ids the host no longer has. | The durable-prefix rule drops the ids above the recovered maximum. |
| After a delete was applied in memory and before a manifest recorded it | The host's records without the record. | Step 6: the liveness check marks the row dead. |
| During a rebuild | The previous persisted state. A rebuild replaces state only at the swap and publishes a new manifest after it. | The previous state opens, or the host rebuilds again. |
| After the host renumbered, before the library heard of it | The host's new numbering and its epoch, if it persisted them with the renumbering (rule 6). Old segments with the old epoch. | The epoch differs, so the index is cold and the host rebuilds. |
| In the middle of a replay | Nothing new. A replay changes nothing durable until the next publish. | The same replay again. It is idempotent. |
| A segment or manifest is damaged | Whatever damage the file has. | `CorruptStateException`, and the host calls `rebuildFrom`. A second failure is a host error. |

## Open questions for a host

A host answers these before it relies on the protocol. A different answer changes a rule above.

1. Can the host persist an epoch atomically with any renumbering of its ids? If it cannot, the library has only the content fingerprint, which detects a renumbering with high probability and does not prove it, and the host must accept a rebuild after every compaction, or a memory-only index.
2. Does the host report every path that reissues ids? Compaction is the obvious one. A restore from a backup, an import, a rebuild of the host's own tables, and the installation of a replica's snapshot are others.
3. Are ids minted in the order in which the host applies batches, even with concurrent transactions? The rule needs a strictly increasing order, so the host serializes `apply` per index and mints in that order, or it mints under a lock that gives the same order.
4. Can the host list its live records above a given id without scanning everything, and test one id for liveness in constant time? The cost of step 6 and step 7 is the cost of these two calls.
5. Can the host tell the highest id it recovered after a crash? Under which flush modes can a commit that it counted as final be lost?
6. Does the host call `apply` only when changes stand? If it must apply earlier, it must undo an add with a delete and a delete with an add of the vector, and the persisted state then may contain ids of transactions that did not stand, which only the durable-prefix rule covers.
7. When does the host open the index relative to its own recovery? The library needs final records.
8. What does a replica, a restored copy, or a copy from a backup contain? The library's files may be absent or older than the records. Absent files give a cold index.
9. How much memory does the staging of a long transaction hold until it applies? The library does not bound that.

[#5]: https://github.com/humemai/tiervec/issues/5
[#7]: https://github.com/humemai/tiervec/issues/7
[#9]: https://github.com/humemai/tiervec/issues/9
[#10]: https://github.com/humemai/tiervec/issues/10
[#17]: https://github.com/humemai/tiervec/issues/17
