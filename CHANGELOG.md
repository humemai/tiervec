# Changelog

All notable changes to tiervec are in this file. The layout follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The version rule and the steps of a release are in [docs/RELEASING.md](docs/RELEASING.md). Until 1.0, a breaking change to the supported surface ([docs/API.md](docs/API.md)) needs a new minor version.

## Unreleased

### Added

- Benchmark harness: `--data` reads a real embedding set that `scripts/prep_real_dataset.py` converts from parquet (raw float32 files with exact answers), and `--engine jvector` builds and searches the same data with JVector at the same `M` and `efConstruction`, in the bench profile only (JVector is not a dependency of the library). A CI smoke runs both on a small generated set ([docs/BENCHMARKS.md](docs/BENCHMARKS.md), [issue #108](https://github.com/humemai/tiervec/issues/108)).
- Benchmark harness: `--quant int8` runs JVector on int8 vectors, with and without a float32 rerank, and `--quant sq8` builds the tier with SQ8 codes. `--targets` reports latency at matched recall, `--disk` the size of the written index, `--props` checks that a write is found by the next search and that a reused id never answers with its old vector, and `--shares` now works with the JVector arm. The bench profile moves to JVector 4.1.0-rc.1 ([issue #118](https://github.com/humemai/tiervec/issues/118)).

### Fixed

- The fp32 graph kernels of a JVM (`VKern`) sum the lanes in a fixed order, like the flat kernels. Before, the sum was `reduceLanes(ADD)`, whose order depends on the JIT tier, so the same search could return a score one ulp apart before and after compilation. This showed as a rare failure of `SegmentForgedFileTest` on arm64 ([#104](https://github.com/humemai/tiervec/issues/104)). A score can differ from the one of 0.1.0 in the last bit.
- A flipped bit in the major version of a segment header is damage (`CorruptException`, section `HEADER`), so `TieredIndex.open` falls back to the previous generation. Before, the version was read before the header CRC, and one flipped bit there in the newest segment made the whole open fail with `UnsupportedException`. A larger major version with an intact header is still refused ([#116](https://github.com/humemai/tiervec/issues/116)).
- A flipped bit in the major version of a manifest is damage (`CorruptException`), so `TieredIndex.open` falls back to the previous generation and a later `checkpoint` replaces the files. Before, the version was read before the CRC: `open` and every `checkpoint` into the directory failed with `UnsupportedException` ([#123](https://github.com/humemai/tiervec/issues/123)).
- A segment file whose first eight bytes are damaged is refused with `CorruptException` (section `HEADER`) and not `UnsupportedException`, and the message names the file and the field. Before, one flipped bit in the magic of the newest segment made `TieredIndex.open` fail, where a flipped bit in the header, footer or locator made it fall back to the previous generation. A larger major version is still an `UnsupportedException` that is refused, not skipped ([#83](https://github.com/humemai/tiervec/issues/83)).
- `TieredIndex.open` flushes the directory before it deletes the files of an older generation. Before, after a process was killed between the rename of a manifest and the flush that follows it, the next open removed the previous generation while the rename was still unflushed, so a power cut at that point could leave no generation at all ([#125](https://github.com/humemai/tiervec/issues/125)).

## 0.1.1 - 2026-11-13

The library code is unchanged from `0.1.0`: no source file changed. The changes are in the release workflow and the pom.

### Changed

- Creating the GitHub release now publishes to Maven Central by itself and verifies the result (checksums, signatures, the built jars, a resolution from a fresh local repository), so the Publish click in the Central Portal is gone ([docs/RELEASING.md](docs/RELEASING.md#the-publish-run)).

## 0.1.0 - 2026-11-12

The first release. It is the sources of the first experiment, imported and brought to the supported surface in [docs/API.md](docs/API.md). Nothing was released under the old package name. Details are in [docs/STATUS.md](docs/STATUS.md).

### Added

- `FlatIndex`: exact scan with optional SQ8 or binary codes, compaction, `serialize` and `load`.
- `GraphIndex`: a concurrent HNSW graph with delete, `consolidate`, and optional SQ8.
- `IvfIndex`: a standalone inverted-file index with lazy training, background retrain and drift detection.
- `TieredIndex`: an exact tail, sealed graph segments, background merges, tombstones with a version floor, atomic batches (`apply`, `awaitCapacity`), `rebuildFrom`, and an opt-in scan-first plan for filtered searches (`filterScanBelow`).
- Persistence of a settled tier: `TieredIndex.checkpoint` and `TieredIndex.open`, with segment files and a manifest ([docs/SEGMENT-FORMAT.md](docs/SEGMENT-FORMAT.md)), and `RecordSource` for reconciling a reopened tier with the records of a host database ([docs/ID-EPOCH-AND-REPLAY.md](docs/ID-EPOCH-AND-REPLAY.md)).
- Scalar twins of every kernel, used when the Vector API module is missing, and Vector API kernels for aarch64 and for GraalVM native images.
- A GraalVM native image smoke test on Linux (x86-64 and arm64), macOS arm64 and Windows x64.
- CI on JDK 21 and 25, with the fast tests also on arm64 and on Windows x64.
- Community files (CONTRIBUTING, SECURITY, SUPPORT, and the issue and pull request templates) and the platform support table in the README.
- The supported public surface and the stability policy ([docs/API.md](docs/API.md)).
- A benchmark harness for the tier (`src/bench`, [docs/BENCHMARKS.md](docs/BENCHMARKS.md)) with a CI smoke at 10,000 vectors. No result file is published yet.
- Maven Central publishing for `ai.humem:tiervec`: the metadata Central requires in `pom.xml`, a `release` profile (sources jar, javadoc jar, GnuPG signatures, upload to the Central Portal), and the workflow `release-central.yml` with a dry run ([docs/RELEASING.md](docs/RELEASING.md#maven-central)).
- The documentation is published as a release version with the alias `latest` when a version tag is pushed, and `dev` stays the label of `main`.

### Changed

- Package renamed from `com.humemai.vec` to `ai.humem.tiervec` (no other change). The system property that forces the scalar twins was renamed with it, from `humemai.vec.scalar` to `ai.humem.tiervec.scalar`.
