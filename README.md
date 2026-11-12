# tiervec

A tiered vector index for agent-memory databases on the JVM.

**Status: experimental.** The API is not stable, there is no release, and nothing is wired into a database engine yet. No benchmark numbers are published. [What is proven](#what-is-proven) says what the repository shows today, and [docs/STATUS.md](docs/STATUS.md) has the detail.

## What it is

A library of vector-index algorithms meant to live inside a database engine, in the same process, with no network between the engine and the index. The first host is [ArcadeDB](https://github.com/ArcadeData/arcadedb).

The design follows the workload of an agent's memory: many small and medium collections, a steady stream of small writes, deletes and updates, and queries that must see a write right away.

- **Exact tail.** Recent writes sit in a bounded buffer that is scanned exactly, so a write is visible to the next query without waiting for an index build.
- **Sealed segments.** Older vectors live in immutable HNSW graph segments. Below a size threshold no graph is built and every search is an exact scan. Segments are merged in the background and swapped in atomically.
- **Tombstones with a version floor** handle deletes and updates, including ids that a database reuses after a delete.
- **Scalar twins of every kernel** (flat, IVF, and graph), chosen once per JVM, so every index, the graph tier and the tiered index included, also runs without the Vector API module.
- **Pure Java**, no native dependencies. It also builds into a GraalVM native image, where every kernel except hamming (fp32, int8, SQ8, and block) uses the Vector API too (see [Native image](#native-image)).

The repository also holds a standalone IVF (inverted file) index, which is not a tier segment yet. The host engine would own persistence, transactions and recovery: the tier can rebuild all of its derived state from the raw vectors. Serialization exists for the flat index and, as a first cut, for a settled tier: segment files with a manifest, written by `TieredIndex.checkpoint` and read by `TieredIndex.open` ([docs/SEGMENT-FORMAT.md](docs/SEGMENT-FORMAT.md)). [docs/DESIGN.md](docs/DESIGN.md) describes what the code does.

## What is proven

Every row says what the repository itself shows, and links the evidence. "Not measured yet" means no number is published here.

| Question | Where it stands | Evidence |
|---|---|---|
| Does it build and pass its tests? | Yes, on Linux x86-64 (JDK 21 and 25, fast and slow tests), Linux arm64, macOS arm64 and Windows x64 (fast tests only on these three). Native image smoke jobs run on the same four platforms. | [Platform support](#platform-support), [docs/STATUS.md](docs/STATUS.md#what-exists) |
| Sizes the tests use | The tests run on small collections. The long id-reuse test (`RidReuseBigLongTest`) and the long IVF program use 20,000 vectors, and the threaded graph program stays below 67,000. A self-search probe on the graph index ran at 100,000 to 200,000 vectors. | [docs/STATUS.md](docs/STATUS.md#graph-self-search-recall) |
| Speed and recall at scale | Not measured yet. No benchmark numbers are published in this repository, and nothing here is proven at 1,000,000 vectors or at 10,000,000. A harness for these measurements exists, and CI runs it at 10,000 vectors as a smoke test. | [docs/BENCHMARKS.md](docs/BENCHMARKS.md), [docs/STATUS.md](docs/STATUS.md) |
| Concurrent readers and writers | Tested at small sizes: threaded programs of 5 to 60 seconds, batch stress tests, and tests that hold a search inside a merge or a seal. Concurrent writers at scale: not proven. | [docs/STATUS.md](docs/STATUS.md#record-id-reuse-and-tombstones) |
| Deletes, updates, reused ids, atomic batches | Tested, including races between a search and a merge or a rebuild. The races found so far ([#3](https://github.com/humemai/tiervec/issues/3), [#26](https://github.com/humemai/tiervec/issues/26), [#28](https://github.com/humemai/tiervec/issues/28)) have tests. | [docs/STATUS.md](docs/STATUS.md#record-id-reuse-and-tombstones), [docs/STATUS.md](docs/STATUS.md#atomic-batches) |
| Restart and recovery | A settled tier can be checkpointed and reopened. Tests stop the publish after each step (a simulated process kill), flip bits in every byte that a CRC covers, and forge files with valid CRCs. Power loss is not tested. | [docs/SEGMENT-FORMAT.md](docs/SEGMENT-FORMAT.md), [docs/STATUS.md](docs/STATUS.md#id-epoch-and-replay) |
| Native image | Smoke tests build and run an image and compare every answer with an exact scan. They do not cover segment files, `rebuildFrom` or concurrent writers. The image is not published. | [docs/STATUS.md](docs/STATUS.md#what-does-not-exist-yet) |
| Inside a database engine | No. Nothing here is wired into ArcadeDB or any other engine. | [docs/PROVIDER-CONTRACT.md](docs/PROVIDER-CONTRACT.md#status-of-every-guarantee) |
| Release | None. The version is `0.1.0-SNAPSHOT` and the API is not stable. | [docs/STATUS.md](docs/STATUS.md#what-does-not-exist-yet) |

**Known limits**

- Graph search is approximate. At the default `ef` (100), a query equal to an inserted vector is missed at about 1.2e-4 to 4.0e-4 per probe at 100,000 to 200,000 vectors, and the rate rises with the size of the index. A search at `ef` 1000 found 1,254 of the 1,260 misses that were diagnosed. The hardware of this measurement is not recorded. ([details](docs/STATUS.md#graph-self-search-recall))
- A filtered search with a small allowed share (about 0.01) spends most of its cost on a walk that falls back to an exact scan. The opt-in `filterScanBelow` skips the walk, it is off by default, and the range of shares where it pays is not measured yet ([#12](https://github.com/humemai/tiervec/issues/12), [docs/DESIGN.md](docs/DESIGN.md#graph-index)).
- `checkpoint` writes a settled tier only: no tail, no tombstones. Reads copy the file into memory, with no memory mapping ([docs/SEGMENT-FORMAT.md](docs/SEGMENT-FORMAT.md#what-the-first-cut-leaves-out)).
- A search that meets a batch waits for it, and a stream of batches that follow each other faster than one scan takes can starve a search ([docs/STATUS.md](docs/STATUS.md#atomic-batches)).
- In a native image a result can differ from the JVM by rounding. AVX-512 and SVE are not used ([#52](https://github.com/humemai/tiervec/issues/52)).

## Build and test

```
mvn -B -ntp verify            # fast tests: Vector API kernels, then the scalar twins
mvn -B -ntp verify -Pslow     # the long concurrency and stress tests (tag "slow")
```

The tests are test programs, wrapped as JUnit 5 tests. A wrapper runs the program, asserts that no check failed, and asserts that at least the expected number of checks ran. The benchmark harness is not part of the Maven build: [docs/BENCHMARKS.md](docs/BENCHMARKS.md) says how to compile and run it.

## Native image

The library runs in a GraalVM native image. The CI job `native image smoke` builds the scalar image, and the jobs `native image smoke (vector, ...)` build the vector image on x86-64 and arm64 Linux and on macOS arm64 (GraalVM Community 25.0.2). The scalar image also builds on arm64 Linux (`native image smoke (arm64 Linux)`) and on macOS arm64 (`native image smoke (macOS arm64)`). The job `native image smoke (vector, Windows x64)` builds the vector image on Windows x64; there is no scalar image job on Windows.

- In an image every kernel that has a Vector API form uses it through `NKernels` (flat, IVF, and graph): the fp32 `dot` and `l2sq`, the int8 dot, the SQ8 dots, and `block16`. The species is fixed when the image is built: 256 bits on x86-64, 128 bits on aarch64 (where the Vector API has no 32-bit byte species, so the byte kernels load 64 or 128 bits of bytes and split them into parts). The Vector API kernels of the JVM are more than 10 times slower than scalar code in an image, so the image does not use them; `NKernels` is the form that the image compiles to SIMD code.
- `hamming` has no Vector API form and stays scalar, in an image and on a JVM. AVX-512 and SVE are not used (the species stay 256 and 128 bits until measured).
- A result in an image can differ from a result on a JVM by rounding, because the lane sum of `NKernels` has the order of the instruction. Results are still bitwise equal between two calls on equal data within one image.
- To build an image with the Vector API kernels, give the image builder `--add-modules jdk.incubator.vector -H:+VectorAPISupport`. The jar carries `META-INF/native-image/ai.humem/tiervec/native-image.properties`, which sets `--initialize-at-build-time=ai.humem.tiervec.NKernels` for an application that has the jar on its class path; the species must be fixed at build time, or the kernels are slow. `-H:+VectorAPISupport` is an experimental option of GraalVM 25, and it needs GraalVM 25 or newer.
- Without `--add-modules jdk.incubator.vector` the image has the scalar twins for every kernel. `-Dai.humem.tiervec.scalar=true` forces the scalar twins in an image as well, when the property is set before the library is first used (an image takes `-D` options on its command line).
- The smoke program prints informational `SPEED` lines, one per kernel (`dot`, `dotI8`, `dotF8`, `dotF8x4`, and `block16` per block of 16 vectors), with the median time of 1,000,000 calls on 768 dimensions next to the scalar twin. They are not a bar.

## Requirements

- JDK 21 or newer. CI builds and tests with Temurin 21 and 25 on x86-64 Linux, and runs the fast tests on arm64 Linux (JDK 21 and 25) and arm64 macOS (JDK 21). The job `fast tests (Windows x64, JDK 21)` runs the fast tests on Windows x64 (not the slow tests).
- Maven 3.9 or newer.
- The Vector API is an incubator module in these JDKs. Compile with `--add-modules jdk.incubator.vector`, and run with it to get the Vector API kernels; the `pom.xml` sets it for the compiler and for the test JVMs. At run time the module is optional: every kernel (flat, IVF, and graph) falls back to its scalar twin when the module is missing, or when `-Dai.humem.tiervec.scalar=true` is set.
- On a JVM with the Vector API, x86-64 vectorizes the fp32 `dot` and `l2sq`, the int8 dot, the SQ8 dots, and `block16`. On aarch64 (NEON, 128-bit float lanes) the fp32 `dot` and `l2sq` use the JVM kernels and the int8 dot, the SQ8 dots, and `block16` use the 128-bit forms of `NKernels`. `hamming` is scalar everywhere. A JVM with 128-bit lanes that is not aarch64 runs those four as scalar code. AVX-512 and SVE are not used ([#52](https://github.com/humemai/tiervec/issues/52)).
- Nothing else at run time. JUnit 5 is a test dependency.

## Platform support

There are three tiers. For tier 1, each row names the CI job that proves it. A job marked "required" must pass before a pull request can merge into `main`. The other jobs run on every pull request but do not block it. All jobs run on GitHub-hosted runners.

**Tier 1: tested in CI on every change**

| Platform | Run by the job | CI job | Required |
|---|---|---|---|
| Linux x86-64, Temurin 21 | fast tests | `fast tests (JDK 21)` | yes |
| Linux x86-64, Temurin 25 | fast tests | `fast tests (JDK 25)` | yes |
| Workflow hygiene (actions pinned to a commit SHA) | lint | `Ensure SHA pinned actions` | yes |
| Linux x86-64, Temurin 21 and 25 | slow tests (long concurrency and stress) | `slow tests (JDK 21)`, `slow tests (JDK 25)` | no |
| Linux arm64, Temurin 21 and 25 | fast tests | `fast tests (arm64 Linux, JDK 21)`, `fast tests (arm64 Linux, JDK 25)` | no |
| macOS arm64, Temurin 21 | fast tests | `fast tests (arm64 macOS, JDK 21)` | no |
| Windows x64, Temurin 21 | fast tests | `fast tests (Windows x64, JDK 21)` | no |
| Linux x86-64, GraalVM Community 21.0.2 | native image smoke, scalar kernels | `native image smoke` | no |
| Linux arm64, GraalVM Community 21.0.2 | native image smoke, scalar kernels | `native image smoke (arm64 Linux)` | no |
| macOS arm64, GraalVM Community 21.0.2 | native image smoke, scalar kernels | `native image smoke (macOS arm64)` | no |
| Linux x86-64, GraalVM Community 25.0.2 | native image smoke, Vector API kernels | `native image smoke (vector, x86-64 Linux)` | no |
| Linux arm64, GraalVM Community 25.0.2 | native image smoke, Vector API kernels | `native image smoke (vector, arm64 Linux)` | no |
| macOS arm64, GraalVM Community 25.0.2 | native image smoke, Vector API kernels | `native image smoke (vector, macOS arm64)` | no |
| Windows x64, GraalVM Community 25.0.2 | native image smoke, Vector API kernels | `native image smoke (vector, Windows x64)` | no |

The slow tests run on Linux x86-64 only. There is no scalar-kernel native image job on Windows. The native image jobs run only when `src/main`, `src/smoke` or their workflow file change.

**Tier 2: measured.** None yet. No benchmark numbers are published in this repository. The harness is described in [docs/BENCHMARKS.md](docs/BENCHMARKS.md); a row moves here when a result file with its hardware, JDK and commit is committed under `results/`. The `SPEED` lines that the native image jobs print are informational, and the one measurement in the repository (the graph self-search recall in [docs/STATUS.md](docs/STATUS.md#graph-self-search-recall)) does not record its hardware. The shared GitHub runners are not a benchmark machine: a timing from them is a ratio between two things measured in the same run, not an absolute speed. A row moves here when its numbers and its hardware are published.

**Tier 3: best effort.** The code is plain Java 21 and has no native dependency, so it is expected to work, but nothing checks it and a problem there is fixed when it is cheap to do.

- macOS on Intel (x86-64): no CI job.
- Other JDK vendors and other JDK versions than 21 and 25: only Temurin 21 and 25 are tested.
- Other Linux distributions and other CPU features: the CI runners fix them. AVX-512 and SVE are not used ([#52](https://github.com/humemai/tiervec/issues/52)).

**Out:** Windows on ARM, until GraalVM and the libraries around it support it. Nothing here builds or tests it.

## Documentation

The pages below are published at <https://docs.humem.ai/tiervec/dev/> as the `dev` version. It follows `main` and is experimental and unreleased, like the library. The site is built from the Markdown files of this repository ([how to build it](CONTRIBUTING.md#build-the-documentation)).

- [docs/API.md](docs/API.md): the supported public classes, the internal ones, and what may change in a 0.x release.
- [docs/DESIGN.md](docs/DESIGN.md): the architecture and the knobs of the tier configuration.
- [docs/PROVIDER-CONTRACT.md](docs/PROVIDER-CONTRACT.md): what a host database must provide and what the library guarantees, with the status of each line.
- [docs/ID-EPOCH-AND-REPLAY.md](docs/ID-EPOCH-AND-REPLAY.md): how ids are numbered, the epoch rule, and replay after a restart.
- [docs/SEGMENT-FORMAT.md](docs/SEGMENT-FORMAT.md): the segment file, the manifest, the publish order, and reopen, as the first cut implements them.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md): the benchmark harness, how to run it, and what a result file holds.
- [docs/STATUS.md](docs/STATUS.md): what exists, what does not exist yet.
- [docs/ROADMAP.md](docs/ROADMAP.md): what is planned and in what order, with the open issues, and the non-goals.
- [docs/RELEASING.md](docs/RELEASING.md): the version rule and the steps of a release. [CHANGELOG.md](CHANGELOG.md) lists the changes.

## Contributing and support

- [CONTRIBUTING.md](CONTRIBUTING.md): how to build, test and open a pull request.
- [SUPPORT.md](SUPPORT.md): who maintains tiervec and where to ask for help.
- [SECURITY.md](SECURITY.md): how to report a security problem privately.

## License

Apache-2.0, see [LICENSE](LICENSE). Papers and libraries this work builds on are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
