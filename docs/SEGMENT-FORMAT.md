# Segment files and the manifest

This page describes what the code writes and reads today (the first cut of [#9](https://github.com/humemai/tiervec/issues/9)). A tier is written to a directory with `TieredIndex.checkpoint(dir)` and read back with `TieredIndex.open(dir, config)`: its sealed segments, the rows that deletes and replaces have hidden, and the rows of its tail ([#158](https://github.com/humemai/tiervec/issues/158)). One sealed graph can also be written alone with `SealedGraph.serialize(file)` and read with `SealedGraph.load(file)`.

The records of the host stay the source of truth. Everything in the directory is derived data that can be deleted and rebuilt with `TieredIndex.rebuildFrom`. When `open` cannot trust the directory, it throws, and the host rebuilds from its records.

The claim the format makes: a process kill at any point of a publish leaves generation N or generation N+1 of the directory, plus files that the next open removes. No claim is made about power loss (see "What a crash leaves behind").

## Files in an index directory

| File | What it holds |
|---|---|
| `seg-<id>.tvs` | One immutable segment: the arrays of one `SealedGraph`. The id is zero-padded to at least six digits. |
| `tomb-<generation>.tvd` | The tombstone bitmaps of every segment of a generation: the rows that a delete or a replace has hidden. |
| `tail-<generation>.tvt` | The rows of the tail of a generation, the ones that no segment holds. No file when the tail was empty. |
| `manifest-<generation>.tvm` | One generation: the live segments, the tombstone file, the tail file, a configuration fingerprint. The generation is zero-padded to 20 digits. |
| `*.tmp` | A file being written. Never named by a manifest. Removed at open. |
| `LOCK` | An empty file whose operating system lock says which process owns the directory (see "One process per directory"). Never deleted. |

Nothing that a manifest names is ever modified. A new state is a new manifest and, when needed, new segment files.

## One process per directory

`checkpoint` and every `open` take an exclusive lock on `LOCK` before they read or write anything else in the directory, and the tier keeps it until `close`, or until a `checkpoint` into another directory has succeeded. A second process gets `DirectoryLockedException`, which names the directory, and nothing there changes. This includes the `open` overloads with a rebuild source: they do not rebuild, because two owners of one directory would delete each other's files. The tiers of one process share the lock, and it is released when the last of them lets go.

The lock is the operating system's, so it ends with the process, however the process ends, and there is no stale lock to remove. On Windows it can end a moment after the process has: an `open` right after a process was killed can throw `DirectoryLockedException`, and a host that restarts at once tries again (seen in `TierKillLongTest` on the Windows runner; the documentation of `LockFileEx` gives no bound for the release). Do not delete the file while a process uses the directory. No lock is taken on a file system other than the default one, or in a directory that the process cannot write to. Whether the lock holds between machines on a network file system depends on that file system, and nothing here tests it. A version before this one neither takes nor checks the lock.

## Segment file

All numbers are little-endian. The file is, in order:

```
[0, 4096)                 header block
[4096, ...)               sections, each starting at a multiple of the alignment, in table order
[F, F + footer length)    footer: metadata and the section table
[end - 32, end)           locator
```

The header block holds nothing that depends on the section bytes, and the CRCs live in the footer, so a writer streams the file from front to back and never goes back.

### Header block

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `TVSEG001` |
| 8 | 2 | major version |
| 10 | 2 | minor version |
| 12 | 4 | header length (52 in this version; the last 4 bytes of the header are its CRC) |
| 16 | 4 | byte-order probe `0x01020304` |
| 20 | 4 | flags (none defined) |
| 24 | 4 | `dim` |
| 28 | 4 | `stride_floats` (equal to `dim` in this version; rows are not padded) |
| 32 | 8 | `n`, the number of rows |
| 40 | 1 | metric code: 0 cosine, 1 dot, 2 L2 |
| 41 | 1 | id width in bytes (8) |
| 42 | 1 | layer 0 layout (0 is CSR, offsets plus edges) |
| 43 | 1 | reserved |
| 44 | 4 | section alignment (4096) |
| 48 | 4 | CRC32C of bytes 0 to 47 |
| 52 | 4044 | zero fill, not covered by the CRC |

Enums are stored as explicit codes, never as ordinals.

### Sections

Every section is a plain array of one fixed-width type. Each starts at a multiple of the alignment; the bytes between the end of one section and the start of the next are zero and are not covered by any CRC.

| Id | Name | Element | Count | Notes |
|---|---|---|---|---|
| 1 | `IDS` | i64 | `n` | Id of each row, in row order. Required. |
| 2 | `VERS` | i64 | `n` | Version of each row, written by the tier. Optional for a standalone graph. |
| 3 | `BYID` | i32 | `n` | Rows in increasing id order. A load uses it as is and does not sort. Required. |
| 4 | `VEC` | f32 | `n * dim` | The fp32 vectors (unit length when the metric is cosine). Required. |
| 5 | `LEVEL` | u8 | `n` | Top layer of each row. Required. |
| 6 | `L0_OFF` | i32 | `n + 1` | CSR offsets of layer 0. Required. |
| 7 | `L0_EDGES` | i32 | `L0_OFF[n]` | Layer 0 neighbor rows. Required. |
| 8 | `UP_IDX` | i32 | `n` | Index into `UP_OFF` for rows above layer 0, -1 otherwise. Required. |
| 9 | `UP_OFF` | i32 | free | Offsets of the upper layers. Required. |
| 10 | `UP_EDGES` | i32 | free | Neighbor rows of the upper layers. Required. |
| 11 | `SQ8_CODES` | i8 | `n * dim` | Present when the graph has SQ8 codes. |
| 12 | `SQ8_PARAMS` | f32 | `3 * n` | `qbase`, then `qdelta`, then `qnorm`. Present with section 11, never alone. |

Section ids are never reused. The `REQUIRED` flag (bit 0 of the section flags) is set on the sections a reader cannot do without, and matters only for ids a reader does not know.

### Footer

The footer starts with 64 bytes of metadata, then one 40-byte entry per section.

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | entry row of the graph |
| 4 | 4 | top level |
| 8 | 4 | `M` |
| 12 | 4 | `efConstruction` |
| 16 | 4 | `efSearch` (a default for searches, not part of the data) |
| 20 | 4 | `alpha` (f32) |
| 24 | 8 | `seed` |
| 32 | 8 | smallest row version (0 when there is no `VERS`) |
| 40 | 8 | largest row version |
| 48 | 4 | flags: bit 0 is the SQ8 parameter flag |
| 52 | 4 | number of sections |
| 56 | 4 | offset of the first section entry (64 in this version) |
| 60 | 4 | reserved |

A section entry is: id u16, flags u16, element width u32, element count u64, offset u64, length u64, CRC32C u32, reserved u32. The CRC covers exactly the section bytes.

### Locator

The last 32 bytes: magic `TVSEGEND` (8), footer offset u64, footer length u32, footer CRC32C u32, total file length u64. The locator is the last thing written. A file that was cut short, or that has bytes appended, does not end in a locator that agrees with its own length, and fails after reading 32 bytes.

### Sizes

The file holds the arrays of the in-memory graph, with `VERS` and `BYID` in place of the two sorted-id arrays the graph keeps, and no deletion bitmap. So, with `VERS` present and the vectors on the heap, the sum of the section lengths equals `SealedGraph.bytesHeld()` minus `8 * ceil(n / 64)` (the bitmap). The file length is `4096`, plus the sections each rounded up to the alignment, plus `64 + 40 * sections`, plus 32. `SegmentFormatTest` checks both for two dimensions, with and without SQ8.

## Integrity rules

- A reader reads the header block, the locator and the footer first, and checks the magic, the header CRC, the version, the byte-order probe, the locator against the real file length, and the footer CRC. This costs the same for a file of any size and happens before any section is read.
- The section table is checked: offsets aligned, ascending, inside the file, element width times count equal to the length, the width and count of every known section as expected, all required sections present.
- `SegmentReader.read(file, true)` also checks the CRC32C of every section and names the section that failed (`CorruptException.section()`). `SealedGraph.load` and `TieredIndex.open` always do. `read(file, false)` skips the section CRCs and is meant for inspection.
- The stored id order is checked while it is used (rows in range, ids strictly increasing), and the layer 0 offsets are checked against the edge count.
- A magic that is not `TVSEG001` is damage. The magic is the same in every version (the version is the major and minor fields after it), so a file that does not start with it is refused with `CorruptException` (section `HEADER`), and the message names the file and the field. `TieredIndex.open` treats such a segment like any other damaged one: see "Open".
- A CRC32C detects accidental damage. It is not protection against tampering.

## Versioning and compatibility

A segment file has a format version `major.minor`.

- The header frame is the same in every version, newer major versions included: the magic in bytes 0 to 7, the major and minor version in bytes 8 to 11, the header length at offset 12, and a CRC32C of all header bytes before it in the last 4 bytes of the header, inside the 4096-byte header block. A reader checks that CRC before it believes the version.
- A header whose CRC does not match is damage, whatever its version field says (`CorruptException`, section `HEADER`). A flipped bit in the version field is this case, and `TieredIndex.open` treats the segment like any other damaged one.
- A reader refuses a file with an intact header and a larger major version (`UnsupportedException`). A major version of 0 with an intact header is refused the same way.
- A header whose byte-order probe reads back swapped (`0x04030201`) is a big-endian header and is refused with `UnsupportedException` before the CRC, which does not read back in that case. No single flipped bit turns the probe into its swapped form.
- A reader opens a file with a larger minor version. It ignores a section whose id it does not know, unless the section is marked required, in which case it refuses.
- A newer minor may append header fields (the header length says where the header ends and where its CRC is) and footer metadata (the offset of the first section entry says where the table starts).
- A different byte order, an id width other than 8, a layer 0 layout other than CSR, and a row stride other than `dim` are refused in this version.

The manifest, the tombstone file, and the tail file carry their own `major.minor` under the same rule. The frame of a manifest is the same in every version: the magic, the major and minor version, the body length at offset 12, and the CRC32C of the body in the last 4 bytes of the file. A reader checks the length and that CRC before it believes the version, so a damaged version field is `CorruptException` and only an intact manifest can be refused as newer. The frame of a tail file is that of a segment header. The tombstone file is compared with the length and CRC that the manifest holds before it is read. A manifest, a segment, or a tail file that an older reader does not understand is refused, not skipped: falling back to an older generation would silently serve stale data.

| File | Version written | Versions read |
|---|---|---|
| Segment | 1 | 1 |
| Tombstone file | 1 | 1 |
| Tail file | 1 | 1 |
| Manifest | 2 | 1 and 2 |

Major 2 of the manifest adds the tail fields. A release that reads major 1 only (0.1.0 and 0.1.1) refuses a directory of major 2 with `UnsupportedException`, on `open` and on `checkpoint`, so it never serves segments without their tail and never replaces the directory. That holds also when the tail was empty. A directory of major 1 still opens, with an empty tail, and the next `checkpoint` writes major 2. `OldReleaseTest` checks both directions with the jar of 0.1.1.

## Manifest and tombstone file

`manifest-<generation>.tvm` holds: magic `TVMAN001`, version, the body length, the generation, the epoch and the high-water id of the host (both "none" for a tier that never had an epoch; otherwise the host's epoch and the largest id that a segment or the tail holds), the version counter of the tier, the next free segment id, a fingerprint of the data-affecting configuration (dimension, metric, id width, `M`, `efConstruction`, whether SQ8 is on, the layer 0 layout), the generation, length and CRC of the tombstone file, since major 2 the tail fields (row count u64, file length u64, CRC32C of the rows u32, reserved u32; all 0 for an empty tail), one entry per live segment (id, row count, file length, the header CRC, smallest and largest row version), and a CRC32C of everything before it. File names are built from the ids in the manifest, never read from it.

`tomb-<generation>.tvd` holds: magic `TVTOMB01`, version, the generation, the number of segments, then for each segment its id, its row count and one bit per row (in 64-bit words, row `r` is bit `r & 63` of word `r >>> 6`), and a CRC32C.

A set bit marks a row that a delete or a replace has hidden. The row stays in its segment file, which is never modified, until a merge rewrites the segment. Each generation has its own tombstone file, so a checkpoint that only deleted ids writes a new tombstone file and a new manifest and no segment. A reader refuses (`CorruptException`, section `TOMBSTONES`) a bit at or above the row count of its segment, and a file that leaves a row of an id visible while it marks a newer row of the same id: a tier hides older copies only.

Version 0.1.0 and 0.1.1 write only zero bits.

## Tail file

`tail-<generation>.tvt` holds the rows that the tier held outside its segments when `checkpoint` ran: the active tail, and a frozen tail that a failed seal left behind. All numbers are little-endian.

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `TVTAIL01` |
| 8 | 2 | major version |
| 10 | 2 | minor version |
| 12 | 4 | header length (40 in this version; a newer minor may append fields) |
| 16 | 8 | generation |
| 24 | 8 | `n`, the number of rows, at least 1 |
| 32 | 4 | `dim` |
| 36 | 4 | CRC32C of the header bytes before it (in the last 4 bytes of the header) |
| 40 | `8 * n` | id of each row (i64) |
| | `8 * n` | version of each row (i64), strictly increasing: the rows are in the order in which they arrived |
| | `4 * n * dim` | the vector of each row (f32), as the host passed it, not normalized |

The file ends with the last vector, so its length is `40 + n * (16 + 4 * dim)`. It has no checksum of its own after the rows: the manifest holds the CRC32C of every byte after the header. (A CRC32C over bytes followed by their own CRC32C is the same constant for any bytes, so a manifest checksum over such a file would not tie the manifest to its content.)

A tail row is the only visible copy of its id. Every copy of that id in a segment is older and marked in the tombstone file: that covers a replace of a sealed id, and an id that was deleted and added again. A deleted id has no tail row. A reader refuses (`CorruptException`, section `TAIL`) a file of another length, header, or row checksum than the manifest says, an id twice, versions that do not increase from at least 1 to at most the version counter, an id of which a segment shows a copy or holds a newer one, and inside an epoch an id above the high-water id.

Every `checkpoint` writes the whole tail again, `40 + n * (16 + 4 * dim)` bytes and one file flush, and writers wait while it does; searches do not. At dimension 768 that is 6.2 MB for 2,000 rows (the default seal threshold) and 24.7 MB for 8,000 rows (the default back pressure limit, which one batch can exceed). `open` reads the file and puts the rows into the flat tail, on the heap also with `MAPPED` storage, and builds no graph. The time of either is not measured yet.

## Publish

`Manifest.publish` makes a new generation live. The segment files it names are written and flushed first (`SegmentWriter.write` creates a new file and flushes it). Before the first generation of a directory, `checkpoint` makes the directory and the parents it lacks, one level at a time, and flushes each name into its parent; it flushes the name of the deepest directory that was already there too, because a killed `checkpoint` may have made it (when that parent may not be read, the name is left to whoever made the directory).

| Step | Action | Hook point (`FaultHook.Step`) |
|---|---|---|
| 1, 2 | Write the segment files, flush each | `SEGMENTS_WRITTEN` |
| 3a | Write and flush the tombstone file, then the tail file when the tail is not empty | `TAIL_WRITTEN` |
| 3b | Flush the directory | `DIR_SYNCED_AFTER_SEGMENTS` |
| 4 | Write the manifest under a temporary name | `MANIFEST_TMP_WRITTEN` |
| 5 | Flush the temporary manifest | `MANIFEST_TMP_SYNCED` |
| 6 | Rename it to `manifest-<generation>.tvm` | `MANIFEST_RENAMED` |
| 7 | Flush the directory | `DIR_SYNCED_AFTER_RENAME` |
| 8 | The caller switches its in-memory view, then `Manifest.gc` deletes what the new manifest does not name | (none) |

The rename is the only moment a generation becomes visible, and by then every byte it names was already flushed. Generation N is deleted only in step 8, after step 7, so an unfinished N+1 can exist only while N and all its files still exist. The cleanup (step 8, and the one at the end of `open`) flushes the directory before its first delete: an open that finds an N+1 whose process was killed between steps 6 and 7 makes that rename durable before it removes N. A manifest is never overwritten: publishing a generation that exists is an error.

On Windows the directory flush of steps 3 and 7 does nothing, because Java cannot open a directory there (`FileChannel.open` is refused). File contents are still forced with `FileChannel.force`, so the bytes of every file are flushed before the rename. The directory entry (the new name, the rename) is not flushed by the library, so a power loss right after a rename can lose the rename on Windows. A process kill is covered on Windows as on Linux, because the claim above is about process kills only.

### Guarantees per platform

These are the platforms that CI runs. "N or N+1" means that the directory opens as the generation before the publish or the new one, never a mixture.

| | Linux x86-64 | Linux arm64 | macOS arm64 | Windows x64 |
|---|---|---|---|---|
| File contents flushed before the rename (`FileChannel.force`) | yes | yes | yes | yes |
| The rename itself flushed (directory flush) | yes | yes | yes | no |
| A process kill during a publish leaves N or N+1 | yes | yes | yes | yes, and the lock of the killed process can be released a moment later |
| A power loss | no claim | no claim | no claim | no claim, and a rename that has returned can be lost |
| Tested by a child process halted at each publish step, and from the directories a power cut can leave (fast: `TierHaltTest`, `TierPowerCutTest`, `ManifestCrashTest`, `TierReopenTest`) | every change | every change | every change | every change |
| Tested by many seeds and random process kills (slow: `TierHaltLongTest`, `TierKillLongTest`) | slow runs | no | no | slow runs |
| Flush order tested on a simulated file system (`SimulatedFsCrashTest`) | every change | every change | every change | no, there is no directory flush to order |

"Every change" is the fast tests of a pull request. "Slow runs" is a push to `main`, the daily run, and a pull request with the label `ci:slow`. None of these tests cuts the power. With the default heap reader no file is mapped while a merge replaces it; what the mapped reader changes, on Windows above all, is in [Mapped vectors](#mapped-vectors).

### What a crash leaves behind

The table is for a process kill (the page cache survives). A crash after a step leaves:

| Crash after | Visible generation | Left on disk for the next open to remove |
|---|---|---|
| 1 or 2 | N | a complete or partial new segment file |
| 3a or 3b | N | new segment files, the new tombstone file, and the new tail file |
| 4 or 5 | N | and a temporary manifest |
| 6 or 7 | N+1 | the files of generation N, its tail file included |
| 8 | N+1 | whatever the cleanup had not yet deleted |

`TierTombstoneTest` does the steps of the table with deleted, replaced and re-added ids in both generations. `TierHaltTest`, `TierKillLongTest`, `TierPowerCutTest`, and `SimulatedFsCrashTest` do their crashes with rows in the tail, and `TierTailTest` and `TierTailCorruptionTest` cover what a reopen brings back and what damage to a tail file does. `ManifestCrashTest` and `TierReopenTest` run the publish with a hook that throws after each step and check which generation `readLatest` and `open` find, that the segments serve the same results as the in-memory graphs, and that the next open leaves exactly the files of that generation. `TierHaltTest` does the same with a real process: a child JVM halts itself (`Runtime.halt`, so no shutdown hook and no cleanup) at each step of a publish, and the parent opens the directory and compares its answers with those of the child's in-memory index. `TierHaltLongTest` (slow) does that for every step in each of three checkpoints with eight seeds, and `TierKillLongTest` (slow) kills a child that publishes in a loop at random moments (`SIGKILL` on Linux), opens the directory after each kill, and goes on in the same directory with the next child. The slow tests run on JDK 21 and 25. `TierPowerCutTest` starts from the files at each step and takes away what the publish order does not yet make durable at that step (a name that was not flushed, the bytes of the temporary manifest), and it also breaks the order on purpose (the rename reached the disk and the bytes of the manifest, of the new segment or of the tombstone file did not). In every case `open` serves exactly generation N or exactly generation N+1 and leaves exactly its files. None of those checks that the flushes are issued in the right order.

`SimulatedFsCrashTest` does, on a simulated file system. It records every write, flush, rename and delete of three publishes (the third after a merge, so it also deletes files), then builds the directories a crash can leave after each of those changes: a file keeps what was flushed and may keep all, part, none, or a torn mix of what was written later, and each new name, rename or delete since the last flush of its directory may or may not be there, in every combination. Each directory must open as exactly generation N or exactly generation N+1 (nothing before the first one is there), answer as the in-memory index of that generation did, hold only that generation's files afterwards, and take the next publish. The test also removes the flushes one at a time and expects a broken directory: every file flush is needed, so is the flush of the directory after the rename (step 7), and the flush of step 3 is needed for the first generation. In later generations a crash that loses a name because step 3 is missing still opens generation N, which is whole until step 7 has returned. A third run kills the process after each change, opens the directory in the next process, and crashes during and after that open. Two more publish into a directory that `checkpoint` makes itself, once straight through and once retried after a kill while the directory was made.

The claim stays a process kill. The simulated file system is a model: it shows that the library asks for its flushes in an order that holds under it, not what a particular file system or drive does when the power goes, and it does not run on Windows, where the directory cannot be flushed.

## Open

`TieredIndex.open(dir, config)`:

1. `Manifest.readLatest`: list the manifests, newest first. For each, check magic, version, CRC, that every named file exists with the recorded length and a valid header (for a segment also footer and locator), and that the tombstone file matches. The first that passes is current. A candidate that fails is skipped; a segment whose first eight bytes are not the magic fails like one with a bad header CRC. If none passes it throws `CorruptException`; an empty directory throws `NoSuchFileException`; a manifest in a newer format throws `UnsupportedException` at once.
2. Compare the fingerprint with the configuration (dimension, metric, `M`, `efConstruction`, SQ8). A difference throws `MismatchException`.
3. Read the tail file and compare its row checksum with the manifest, then read every segment with every section checksummed and compare it with its manifest entry. With `vectorStorage` set to `MAPPED` the `VEC` section is checksummed and then mapped instead of copied ([Mapped vectors](#mapped-vectors)).
4. Delete the files that the manifest does not name (orphans of a crashed publish, older generations).
5. Restore the version counter, the epoch and the floor map (for each id with a marked row, one more than the largest marked version, which hides exactly the marked rows), and start the tier with the segments and with the tail rows back in the tail.

A damaged row of the tail is found in step 3, like a damaged section of a segment: `open` throws and does not fall back to an older generation. A tail file that is missing, of another length, or with a damaged header is found in step 1.

Any failure throws. The overload `open(dir, dim, metric, config, rebuildSource)` catches it and builds the tier from the records with `rebuildFrom`; `openFallbackCause()` tells why. The directory is left as it was in that case. The next `checkpoint` replaces its contents, with one exception: when `open` failed with `UnsupportedException`, `checkpoint` into that directory throws the same exception and changes nothing, so that an older reader does not destroy files of a newer format. A host that wants to give those files up deletes the directory.

`open(dir, dim, metric, config, recordSource)` adds, after step 5, the epoch comparison (another epoch gives a cold index), the drop of ids above the host's recovered maximum (rebuilt segments and a new generation before it returns), the liveness check, and the replay of the host's live records above the high-water id: see [ID-EPOCH-AND-REPLAY.md](ID-EPOCH-AND-REPLAY.md).

## Mapped vectors

By default `TieredIndex.open` copies every section into heap arrays. With `Config.vectorStorage` set to `MAPPED` it leaves the fp32 vectors (section `VEC`) in the segment file and maps them read-only ([#27](https://github.com/humemai/tiervec/issues/27)). The file format does not change.

```java
TieredIndex.Config config = TieredIndex.Config.defaults().withVectorStorage(TieredIndex.VectorStorage.MAPPED);
TieredIndex index = TieredIndex.open(dir, config);
```

The key of the configuration map is `vectorStorage`. It is not in the fingerprint, and `reconfigure` does not change it.

| | `HEAP` (default) | `MAPPED` |
|---|---|---|
| fp32 vectors of a segment that `open` reads | a heap array of `4 * n * dim` bytes | the mapped file |
| Ids, versions, graph, SQ8 codes and parameters | heap | heap |
| Checks at `open` | every section CRC32C | the same, `VEC` included |
| Hits and scores | | the same |
| A distance to a vector | reads the array | first copies the row into a `float[]` of the searching thread |

**What it saves and costs.** On shared CI runners the used heap after opening one segment fell from 153.8 to 55.8 MiB for 200,000 vectors of dimension 128 (vector section 97.7 MiB), and from 194.2 to 46.3 MiB for 50,000 of dimension 768 (146.5 MiB). On JDK 21 without preview features the kernels read a `float[]`, so every distance on a mapped row copies it first. In that run a walk of fp32 vectors took 0.92 to 1.40 times as long as on the heap, a walk of SQ8 codes, which reads fp32 vectors only to rescore, 0.99 to 1.03 times, and the exact scan 1.10 to 1.54 times. An earlier run measured 1.13 to 1.37 for the fp32 walk. Heap stays the default for that cost. The method, the tables, and the commands are in [BENCHMARKS.md](BENCHMARKS.md#the-cost-of-a-row-copy).

**What it covers.** Only the segments that `open` reads. A segment that a seal, a merge, or a rebuild builds in a running tier has its vectors on the heap, also after a `checkpoint` has written it; a later `open` maps it. So a mapped tier saves less heap the longer it seals and merges. `bytesHeld`, which `HostEnv.memoryBudgetBytes` is compared with, leaves the mapped vectors out. The pages in use are in the page cache of the operating system.

**Checks.** `open` reads the `VEC` section once through the file channel to check its CRC32C, so the open still reads every byte. It maps the section only after every check of the segment has passed: a segment that is refused is not mapped. The first searches can have to fault the pages of the mapping in.

**Sections above 2 GiB.** One mapping holds at most 2^31 - 1 bytes. A larger section is mapped in several pieces, each a power of two of whole rows, so no row crosses a piece. The tests lower the limit of a piece to cover this with small files; no test writes a section above 2 GiB. The limit of the format, `n * dim` at most 2^31 - 9, holds for both readers.

**Unmapping.** The library never unmaps. A search can still be reading a segment that a merge has replaced, or one of a tier that was closed meanwhile, and a read of an unmapped page ends the JVM. The JVM unmaps a piece after the garbage collector has found it unreachable, and nothing bounds when that happens. `close` releases the directory lock and not the mappings, so drop the references to a closed tier.

**Deleting a mapped file.** A `checkpoint` or an `open` can retire a segment whose file is still mapped.

- Linux and macOS delete the file as usual, and free its disk space only when the mapping is gone.
- Windows: on the CI runner (Windows Server 2025, Temurin 21) the retired file was deleted while it was still mapped, and the mapping went on serving its bytes (`MappedVectorsTest`, the line `MAPPED-DELETE`). Windows is also known to refuse such a delete. When it does, `checkpoint` and `open` leave the file and go on, and a later one tries again. No test reaches that path, because the runner allows the delete, and nothing here says which Windows versions or file systems refuse it.

**The files must not change.** Nothing may modify, truncate, or overwrite a file that is mapped; the library itself never modifies a segment file. The CRC32C is checked at `open` only, so a later change reaches searches unnoticed, and a page that cannot be loaded (a file that was cut short, a read error of the device) can throw an `InternalError` out of a search or end the JVM. A heap tier does not depend on its segment files after `open`. Use `HEAP` for a directory on a network file system.

**Tests.** The round-trip and corruption tests run with both readers: `MappedSegmentFormatTest`, `MappedSegmentCorruptionTest`, `MappedSegmentForgedFileTest`, `MappedManifestCrashTest`, `MappedTierReopenTest`, and `MappedTierCorruptionTest` run the tests of their parent classes with the mapped reader. They do not run on Windows: they rewrite and delete segment files right after a load, while a file can still be mapped, and that was not checked there. `MappedVectorsTest` runs on every platform: rows over several mappings, one scratch array per thread, a mapped tier that goes on writing and merging, a refused file, and a retired file that is still mapped.

## What the first cut leaves out

- Memory mapping of anything but the vectors. Only `VEC` can be mapped ([Mapped vectors](#mapped-vectors)); the other sections are read into heap arrays, and so is every segment that a running tier builds.
- A log of the writes between two checkpoints. The library writes at `checkpoint` only; the host replays what a generation lacks through `RecordSource`.
- An incremental tail: every `checkpoint` writes all of it.
- Soft-deleted rows inside a segment file: `SegmentWriter` refuses a graph that has them. The tier does not make such graphs; it records hidden rows in the tombstone file.
- Lazy and chunked CRC verification. Every section is verified at open.
- The fixed-stride layer 0 layout and padded rows.
- Formats for the flat tail and for IVF.
- A marker for an unclean shutdown.
- A claim about power loss. The order of the flushes is tested on a simulated file system only.
- Per-segment repair: any integrity failure rebuilds the whole tier.
