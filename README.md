# tiervec

A tiered vector index for agent-memory databases on the JVM.

**Status: experimental and private.** The sources of the first experiment are imported and build with Maven, and their tests run in CI on JDK 21 and 25, and the fast tests also on arm64 (Linux and macOS) and on Windows x64. There is no release, the API is not stable, nothing is wired into a database engine yet, and no measurements are published. See [docs/STATUS.md](docs/STATUS.md).

## What it is

A library of vector-index algorithms meant to live inside a database engine, in the same process, with no network between the engine and the index. The first host is [ArcadeDB](https://github.com/ArcadeData/arcadedb).

The design follows the workload of an agent's memory: many small and medium collections, a steady stream of small writes, deletes and updates, and queries that must see a write right away.

- **Exact tail.** Recent writes sit in a bounded buffer that is scanned exactly, so a write is visible to the next query without waiting for an index build.
- **Sealed segments.** Older vectors live in immutable HNSW graph segments. Below a size threshold no graph is built and every search is an exact scan. Segments are merged in the background and swapped in atomically.
- **Tombstones with a version floor** handle deletes and updates, including ids that a database reuses after a delete.
- **Scalar twins of every kernel** (flat, IVF, and graph), chosen once per JVM, so every index, the graph tier and the tiered index included, also runs without the Vector API module.
- **Pure Java**, no native dependencies. It also builds into a GraalVM native image, where every kernel except hamming (fp32, int8, SQ8, and block) uses the Vector API too (see [Native image](#native-image)).

The repository also holds a standalone IVF (inverted file) index, which is not a tier segment yet. The host engine would own persistence, transactions and recovery: the tier can rebuild all of its derived state from the raw vectors. Serialization exists for the flat index and, as a first cut, for a settled tier: segment files with a manifest, written by `TieredIndex.checkpoint` and read by `TieredIndex.open` ([docs/SEGMENT-FORMAT.md](docs/SEGMENT-FORMAT.md)). [docs/DESIGN.md](docs/DESIGN.md) describes what the code does.

## Build and test

```
mvn -B -ntp verify            # fast tests: Vector API kernels, then the scalar twins
mvn -B -ntp verify -Pslow     # the long concurrency and stress tests (tag "slow")
```

The tests are the test programs of the experiment, wrapped as JUnit 5 tests. A wrapper runs the program, asserts that no check failed, and asserts that at least the expected number of checks ran.

## Native image

The library runs in a GraalVM native image. The CI job `native image smoke` builds the scalar image, and the jobs `native image smoke (vector, ...)` build the vector image on x86-64 and arm64 Linux and on macOS arm64 (GraalVM Community 25.0.2). The scalar image also builds on arm64 Linux (`native image smoke (arm64 Linux)`) and on macOS arm64 (`native image smoke (macOS arm64)`). The job `native image smoke (vector, Windows x64)` builds the vector image on Windows x64; there is no scalar image job on Windows.

- In an image every kernel that has a Vector API form uses it through `NKernels` (flat, IVF, and graph): the fp32 `dot` and `l2sq`, the int8 dot, the SQ8 dots, and `block16`. The species is fixed when the image is built: 256 bits on x86-64, 128 bits on aarch64 (where the Vector API has no 32-bit byte species, so the byte kernels load 64 or 128 bits of bytes and split them into parts). The Vector API kernels of the JVM are more than 10 times slower than scalar code in an image, so the image does not use them; `NKernels` is the form that the image compiles to SIMD code.
- `hamming` has no Vector API form and stays scalar, in an image and on a JVM. AVX-512 and SVE are not used (the species stay 256 and 128 bits until measured).
- A result in an image can differ from a result on a JVM by rounding, because the lane sum of `NKernels` has the order of the instruction. Results are still bitwise equal between two calls on equal data within one image.
- To build an image with the Vector API kernels, give the image builder `--add-modules jdk.incubator.vector -H:+VectorAPISupport`. The jar carries `META-INF/native-image/com.humemai/tiervec/native-image.properties`, which sets `--initialize-at-build-time=com.humemai.vec.NKernels` for an application that has the jar on its class path; the species must be fixed at build time, or the kernels are slow. `-H:+VectorAPISupport` is an experimental option of GraalVM 25, and it needs GraalVM 25 or newer.
- Without `--add-modules jdk.incubator.vector` the image has the scalar twins for every kernel. `-Dhumemai.vec.scalar=true` forces the scalar twins in an image as well, when the property is set before the library is first used (an image takes `-D` options on its command line).
- The smoke program prints informational `SPEED` lines, one per kernel (`dot`, `dotI8`, `dotF8`, `dotF8x4`, and `block16` per block of 16 vectors), with the median time of 1,000,000 calls on 768 dimensions next to the scalar twin. They are not a bar.

## Requirements

- JDK 21 or newer. CI builds and tests with Temurin 21 and 25 on x86-64 Linux, and runs the fast tests on arm64 Linux (JDK 21 and 25) and arm64 macOS (JDK 21). The job `fast tests (Windows x64, JDK 21)` runs the fast tests on Windows x64 (not the slow tests).
- Maven 3.9 or newer.
- The Vector API is an incubator module in these JDKs. Compile with `--add-modules jdk.incubator.vector`, and run with it to get the Vector API kernels; the `pom.xml` sets it for the compiler and for the test JVMs. At run time the module is optional: every kernel (flat, IVF, and graph) falls back to its scalar twin when the module is missing, or when `-Dhumemai.vec.scalar=true` is set.
- On a JVM with the Vector API, x86-64 vectorizes the fp32 `dot` and `l2sq`, the int8 dot, the SQ8 dots, and `block16`. On aarch64 (NEON, 128-bit float lanes) the fp32 `dot` and `l2sq` use the JVM kernels and the int8 dot, the SQ8 dots, and `block16` use the 128-bit forms of `NKernels`. `hamming` is scalar everywhere. A JVM with 128-bit lanes that is not aarch64 runs those four as scalar code. AVX-512 and SVE are not used ([#52](https://github.com/humemai/tiervec/issues/52)).
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
