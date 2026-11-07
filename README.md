# tiervec

A tiered vector index for agent-memory databases on the JVM.

**Status: experimental and private.** The first measurements are not published yet, and nothing here is usable.

## What it is

A library of vector-index algorithms meant to live inside a database engine, in the same process, with no network between the engine and the index. The first host is [ArcadeDB](https://github.com/ArcadeData/arcadedb).

The design follows the workload of an agent's memory: many small and medium collections, a steady stream of small writes, deletes and updates, and queries that must see a write right away.

- **Exact tail.** Recent writes sit in a bounded buffer that is scanned exactly, so a write is visible to the next query without waiting for an index build.
- **Sealed segments.** Older vectors live in immutable segments: a quantized flat scan for small sizes, an HNSW graph above a size threshold, and an inverted-file (IVF) tier for large, cold data. Segments are merged in the background and swapped in atomically.
- **Tombstones with a version floor** handle deletes and updates, including ids that a database reuses after a delete.
- **A scalar twin of every SIMD kernel**, chosen at run time, so the same code runs on the JVM and in a GraalVM native image.
- **Pure Java**, no native dependencies.

The host engine owns persistence, transactions and recovery. tiervec provides the kernels, the graph, IVF and quantizers, serialization, and the tier logic around them.

## License

Apache-2.0, see [LICENSE](LICENSE). Papers and libraries this work builds on are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
