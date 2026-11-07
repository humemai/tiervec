# tiervec

A tiered vector index for agent-memory databases on the JVM.

**Status: experimental and private.** The sources of the first experiment are imported and build with Maven, and their tests run in CI on JDK 21 and 25. There is no release, the API is not stable, nothing is wired into a database engine yet, and no measurements are published. See [docs/STATUS.md](docs/STATUS.md).

## What it is

A library of vector-index algorithms meant to live inside a database engine, in the same process, with no network between the engine and the index. The first host is [ArcadeDB](https://github.com/ArcadeData/arcadedb).

The design follows the workload of an agent's memory: many small and medium collections, a steady stream of small writes, deletes and updates, and queries that must see a write right away.

- **Exact tail.** Recent writes sit in a bounded buffer that is scanned exactly, so a write is visible to the next query without waiting for an index build.
- **Sealed segments.** Older vectors live in immutable HNSW graph segments. Below a size threshold no graph is built and every search is an exact scan. Segments are merged in the background and swapped in atomically.
- **Tombstones with a version floor** handle deletes and updates, including ids that a database reuses after a delete.
- **Scalar twins of the flat-tier kernels**, chosen once per JVM, so the flat and IVF indexes also run without the Vector API module. The graph kernels use the Vector API directly and have no scalar twin yet.
- **Pure Java**, no native dependencies.

The repository also holds a standalone IVF (inverted file) index, which is not a tier segment yet. The host engine would own persistence, transactions and recovery: the tier can rebuild all of its derived state from the raw vectors. Serialization exists for the flat index and, as a first cut, for a settled tier: segment files with a manifest, written by `TieredIndex.checkpoint` and read by `TieredIndex.open` ([docs/SEGMENT-FORMAT.md](docs/SEGMENT-FORMAT.md)). [docs/DESIGN.md](docs/DESIGN.md) describes what the code does.

## Build and test

```
mvn -B -ntp verify            # fast tests: Vector API kernels, then the scalar twins
mvn -B -ntp verify -Pslow     # the long concurrency and stress tests (tag "slow")
```

The tests are the test programs of the experiment, wrapped as JUnit 5 tests. A wrapper runs the program, asserts that no check failed, and asserts that at least the expected number of checks ran.

## Requirements

- JDK 21 or newer. CI builds and tests with Temurin 21 and 25.
- Maven 3.9 or newer.
- The Vector API is an incubator module in these JDKs. Compile and run with `--add-modules jdk.incubator.vector`; the `pom.xml` sets it for the compiler and for the test JVMs. The graph and tier code need the module at run time. The flat and IVF kernels fall back to scalar code when it is missing, or when `-Dhumemai.vec.scalar=true` is set.
- Nothing else at run time. JUnit 5 is a test dependency.

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md): the architecture and the knobs of the tier configuration.
- [docs/PROVIDER-CONTRACT.md](docs/PROVIDER-CONTRACT.md): what a host database must provide and what the library guarantees, with the status of each line.
- [docs/ID-EPOCH-AND-REPLAY.md](docs/ID-EPOCH-AND-REPLAY.md): how ids are numbered, the epoch rule, and replay after a restart.
- [docs/SEGMENT-FORMAT.md](docs/SEGMENT-FORMAT.md): the segment file, the manifest, the publish order, and reopen, as the first cut implements them.
- [docs/STATUS.md](docs/STATUS.md): what exists, what does not exist yet.
- [docs/ROADMAP.md](docs/ROADMAP.md): what is planned and in what order, with the open issues, and the non-goals.

## License

Apache-2.0, see [LICENSE](LICENSE). Papers and libraries this work builds on are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
