# Public API and stability policy

This page says what a host may depend on and what may change. It describes the code on `main` (version `0.1.0-SNAPSHOT`). There is no release yet.

All classes live in one Java package, `ai.humem.tiervec`, and every class that a host needs is public. Some other classes are public only because the classes of the library call each other. This page separates the two groups. The package has the name of the library, and the Maven coordinates are `ai.humem:tiervec`.

## Supported classes

The supported surface is these classes, their public members as listed here, and the behavior that [PROVIDER-CONTRACT.md](PROVIDER-CONTRACT.md) marks `exists today`.

| Class | What it is | Supported members |
|---|---|---|
| `VecIndex` | The interface of every index | All |
| `Hit` | One result: an id and a score, higher is better | All |
| `Metric` | `COSINE`, `DOT`, `L2` | All |
| `TieredIndex` | The index a host database uses: exact tail, sealed segments, merges, batches, checkpoint and reopen | The constructors, `add`, `delete`, `search` (all overloads), `apply`, `awaitCapacity`, `filterScanBelow`, `sealNow`, `mergeNow`, `awaitIdle`, `rebuildFrom` (all overloads), `epoch`, `checkpoint`, `open` (all overloads), `openFallbackCause`, `config`, `stats`, `segmentSizes`, `lastError`, `size`, `dim`, `metric`, `bytesHeld`, `close` |
| `TieredIndex.Config`, `TieredIndex.Stats`, `TieredIndex.Batch`, `TieredIndex.IdVector`, `TieredIndex.Snapshot`, `TieredIndex.SearchResult`, `TieredIndex.Freshness` | The records and the enum of the tier | All |
| `RecordSource` | What a host offers so that `TieredIndex.open` can reconcile persisted state with the host's records | All |
| `FlatIndex`, `GraphIndex`, `IvfIndex` | Standalone indexes: exact and quantized scan, concurrent HNSW, inverted file | The constructors, the methods of `VecIndex`, the nested `Params` of `GraphIndex` and `IvfIndex`, and the maintenance and diagnostic methods that each class documents in its Javadoc |
| `Quant`, `Sq8Kernel` | The code kind and the SQ8 kernel that the flat index and the tier configuration name | All |
| `SegmentFormat.CorruptException`, `SegmentFormat.UnsupportedException`, `SegmentFormat.MismatchException` | What `TieredIndex.open` throws for damage, for a newer format, and for files of another index definition | The three nested classes. The rest of `SegmentFormat` is internal |

Public members of the supported classes that are not in the table are for tests and diagnostics. They can change in any release: `TieredIndex.setSealHooks`, `activity`, `workerAlive`, `segmentCount`, `pendingTail`, `sealRatePerSec` and `setSealRatePerSec`; `GraphIndex.seal` and `GraphIndex.fromSealed`, which use the internal type `SealedGraph`.

The maintenance and diagnostic methods of the standalone indexes (for example `IvfIndex.listSkew`, `FlatIndex.slotsUsed`) are supported, but the three standalone indexes are the least settled part of the surface. A change to one of them is listed in the changelog like any other.

## Internal classes

These classes are public but not supported. They carry the annotation `@Internal`. A host must not use them. They can change or disappear in any release, with no changelog line.

| Class | What it is |
|---|---|
| `FaultHook` | A seam for tests that stops a publish after a step |
| `Manifest` | The manifest file of a checkpoint |
| `SealedGraph` | An immutable graph segment, in memory and as a file |
| `SegmentFormat` | The constants and helpers of the segment file (except the three exception classes above) |
| `SegmentInfo` | The facts that a segment file header holds |
| `SegmentReader` | Reads a segment file |
| `SegmentWriter` | Writes a segment file |

The test `ApiSurfaceTest` checks this page against the code: every public top-level class is either in the supported table or marked `@Internal`, and the two lists here match the annotation. All other classes of the package are package-private.

A module descriptor that exports only the supported classes is not possible while the internal classes share the package with them. Moving them to a separate package would be a later, breaking change.

## Stability policy for 0.x

The version is `0.MINOR.PATCH` until the API is stable (see the plan in [issue 61](https://github.com/humemai/tiervec/issues/61)).

- A breaking change to the supported surface needs a new minor version, and a line in `CHANGELOG.md` that says what to change.
- A patch release never makes a breaking change to the supported surface.
- Breaking means: a supported class or member is removed, renamed or moved; a signature, a return type or a thrown checked exception changes; a record gets a new component (its constructor changes, unless the old constructor stays, as `TieredIndex.Config` does for its earlier form); or behavior changes against a guarantee that [PROVIDER-CONTRACT.md](PROVIDER-CONTRACT.md) marks `exists today`.
- Not breaking: a new class, a new method, a new overload, or a new constant in an enum. An added enum constant is listed in the changelog.
- A guarantee marked `partly` or `planned` in the contract can change in any minor release.
- The internal classes and the members that are not in the table above can change in any release.
- The file format has its own version. Read [SEGMENT-FORMAT.md](SEGMENT-FORMAT.md#versioning-and-compatibility) for what a reader accepts. A change to the format that an older reader refuses is listed in the changelog.
- At 1.0 the rule tightens: a breaking change then needs a new major version.

The Javadoc of the supported classes says what each method does, and it is the reference for the details.
