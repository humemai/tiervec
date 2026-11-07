# Segment files and the manifest

This page describes what the code writes and reads today (the first cut of [#9](https://github.com/humemai/tiervec/issues/9)). A settled tier is written to a directory with `TieredIndex.checkpoint(dir)` and read back with `TieredIndex.open(dir, config)`. One sealed graph can also be written alone with `SealedGraph.serialize(file)` and read with `SealedGraph.load(file)`.

The records of the host stay the source of truth. Everything in the directory is derived data that can be deleted and rebuilt with `TieredIndex.rebuildFrom`. When `open` cannot trust the directory, it throws, and the host rebuilds from its records.

The claim the format makes: a process kill at any point of a publish leaves generation N or generation N+1 of the directory, plus files that the next open removes. No claim is made about power loss (see "What a crash leaves behind").

## Files in an index directory

| File | What it holds |
|---|---|
| `seg-<id>.tvs` | One immutable segment: the arrays of one `SealedGraph`. The id is zero-padded to at least six digits. |
| `tomb-<generation>.tvd` | The tombstone bitmaps of every segment of a generation. In this version they are all zero. |
| `manifest-<generation>.tvm` | One generation: the live segments, the tombstone file, a configuration fingerprint. The generation is zero-padded to 20 digits. |
| `*.tmp` | A file being written. Never named by a manifest. Removed at open. |

Nothing that a manifest names is ever modified. A new state is a new manifest and, when needed, new segment files.

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

The file holds the arrays of the in-memory graph, with `VERS` and `BYID` in place of the two sorted-id arrays the graph keeps, and no deletion bitmap. So, with `VERS` present, the sum of the section lengths equals `SealedGraph.bytesHeld()` minus `8 * ceil(n / 64)` (the bitmap). The file length is `4096`, plus the sections each rounded up to the alignment, plus `64 + 40 * sections`, plus 32. `SegmentFormatTest` checks both for two dimensions, with and without SQ8.

## Integrity rules

- A reader reads the header block, the locator and the footer first, and checks the magic, the version, the byte-order probe, the header CRC, the locator against the real file length, and the footer CRC. This costs the same for a file of any size and happens before any section is read.
- The section table is checked: offsets aligned, ascending, inside the file, element width times count equal to the length, the width and count of every known section as expected, all required sections present.
- `SegmentReader.read(file, true)` also checks the CRC32C of every section and names the section that failed (`CorruptException.section()`). `SealedGraph.load` and `TieredIndex.open` always do. `read(file, false)` skips the section CRCs and is meant for inspection.
- The stored id order is checked while it is used (rows in range, ids strictly increasing), and the layer 0 offsets are checked against the edge count.
- A CRC32C detects accidental damage. It is not protection against tampering.

## Versioning and compatibility

A segment file has a format version `major.minor`.

- A reader refuses a file with a larger major version (`UnsupportedException`).
- A reader opens a file with a larger minor version. It ignores a section whose id it does not know, unless the section is marked required, in which case it refuses.
- A newer minor may append header fields (the header length says where the header ends and where its CRC is) and footer metadata (the offset of the first section entry says where the table starts).
- A different byte order, an id width other than 8, a layer 0 layout other than CSR, and a row stride other than `dim` are refused in this version.

The manifest and the tombstone file carry their own `major.minor` under the same rule. A manifest or a segment that an older reader does not understand is refused, not skipped: falling back to an older generation would silently serve stale data.

## Manifest and tombstone file

`manifest-<generation>.tvm` holds: magic `TVMAN001`, version, the body length, the generation, the epoch and the high-water id of the host (both "none" in this version), the version counter of the tier, the next free segment id, a fingerprint of the data-affecting configuration (dimension, metric, id width, `M`, `efConstruction`, whether SQ8 is on, the layer 0 layout), the generation, length and CRC of the tombstone file, one entry per live segment (id, row count, file length, the header CRC, smallest and largest row version), and a CRC32C of everything before it. File names are built from the ids in the manifest, never read from it.

`tomb-<generation>.tvd` holds: magic `TVTOMB01`, version, the generation, the number of segments, then for each segment its id, its row count and one bit per row, and a CRC32C. This version writes only zero bits and refuses a file with a bit set.

## Publish

`Manifest.publish` makes a new generation live. The segment files it names are written and flushed first (`SegmentWriter.write` creates a new file and flushes it).

| Step | Action | Hook point (`FaultHook.Step`) |
|---|---|---|
| 1, 2 | Write the segment files, flush each | `SEGMENTS_WRITTEN` |
| 3 | Write and flush the tombstone file, flush the directory | `DIR_SYNCED_AFTER_SEGMENTS` |
| 4 | Write the manifest under a temporary name | `MANIFEST_TMP_WRITTEN` |
| 5 | Flush the temporary manifest | `MANIFEST_TMP_SYNCED` |
| 6 | Rename it to `manifest-<generation>.tvm` | `MANIFEST_RENAMED` |
| 7 | Flush the directory | `DIR_SYNCED_AFTER_RENAME` |
| 8 | The caller switches its in-memory view, then `Manifest.gc` deletes what the new manifest does not name | (none) |

The rename is the only moment a generation becomes visible, and by then every byte it names was already flushed. Generation N is deleted only in step 8, after step 7, so an unfinished N+1 can exist only while N and all its files still exist. A manifest is never overwritten: publishing a generation that exists is an error.

### What a crash leaves behind

The table is for a process kill (the page cache survives). A crash after a step leaves:

| Crash after | Visible generation | Left on disk for the next open to remove |
|---|---|---|
| 1 or 2 | N | a complete or partial new segment file |
| 3 | N | new segment files and the new tombstone file |
| 4 or 5 | N | and a temporary manifest |
| 6 or 7 | N+1 | the files of generation N |
| 8 | N+1 | whatever the cleanup had not yet deleted |

`ManifestCrashTest` and `TierReopenTest` run the publish with a hook that throws after each step and check which generation `readLatest` and `open` find, that the segments serve the same results as the in-memory graphs, and that the next open leaves exactly the files of that generation. This covers a process kill only. It does not check that the flushes are issued in the right order, and it says nothing about power loss: that needs a simulated file system that discards data that was not flushed, which this version does not have.

## Open

`TieredIndex.open(dir, config)`:

1. `Manifest.readLatest`: list the manifests, newest first. For each, check magic, version, CRC, that every named file exists with the recorded length and a valid header, footer and locator, and that the tombstone file matches. The first that passes is current. A candidate that fails is skipped. If none passes it throws `CorruptException`; an empty directory throws `NoSuchFileException`; a manifest in a newer format throws `UnsupportedException` at once.
2. Compare the fingerprint with the configuration (dimension, metric, `M`, `efConstruction`, SQ8). A difference throws `MismatchException`.
3. Read every segment with every section checksummed, and compare it with its manifest entry.
4. Delete the files that the manifest does not name (orphans of a crashed publish, older generations).
5. Restore the version counter and start the tier with the segments, an empty tail and an empty floor map.

Any failure throws. The overload `open(dir, dim, metric, config, rebuildSource)` catches it and builds the tier from the records with `rebuildFrom`; `openFallbackCause()` tells why. The directory is left as it was in that case, and the next `checkpoint` replaces its contents.

## What the first cut leaves out

- Memory mapping. A segment is read into the heap arrays of `SealedGraph`.
- Replay of the tail and the high-water id. The manifest carries the epoch and the high-water id as "none". The host replays whatever it needs after `open`.
- Tombstones. `checkpoint` refuses a tier with a non-empty tail, a non-empty floor map, or a segment with hidden copies (an `IllegalStateException`), and `SegmentWriter` refuses a graph with soft-deleted rows. The tombstone file exists and is verified, but holds only zeros. Rebuilding the floor map from bitmaps is the follow-up.
- Lazy and chunked CRC verification. Every section is verified at open.
- The fixed-stride layer 0 layout and padded rows.
- Formats for the flat tail and for IVF.
- A lock file and a marker for an unclean shutdown. One process must own a directory.
- A simulated file system for power loss tests.
- Per-segment repair: any integrity failure rebuilds the whole tier.
