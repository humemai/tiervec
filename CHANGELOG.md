# Changelog

All notable changes to tiervec are in this file. The layout follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The version rule and the steps of a release are in [docs/RELEASING.md](docs/RELEASING.md). Until 1.0, a breaking change to the supported surface ([docs/API.md](docs/API.md)) needs a new minor version.

## Unreleased

There is no release yet. This section holds what the first release (`0.1.0`) will contain, as the repository has it today. Details are in [docs/STATUS.md](docs/STATUS.md).

### Added

- `FlatIndex`: exact scan with optional SQ8 or binary codes, compaction, `serialize` and `load`.
- `GraphIndex`: a concurrent HNSW graph with delete, `consolidate`, and optional SQ8.
- `IvfIndex`: a standalone inverted-file index with lazy training, background retrain and drift detection.
- `TieredIndex`: an exact tail, sealed graph segments, background merges, tombstones with a version floor, atomic batches (`apply`, `awaitCapacity`), `rebuildFrom`, and an opt-in scan-first plan for filtered searches (`filterScanBelow`).
- Persistence of a settled tier: `TieredIndex.checkpoint` and `TieredIndex.open`, with segment files and a manifest ([docs/SEGMENT-FORMAT.md](docs/SEGMENT-FORMAT.md)), and `RecordSource` for reconciling a reopened tier with the records of a host database ([docs/ID-EPOCH-AND-REPLAY.md](docs/ID-EPOCH-AND-REPLAY.md)).
- Scalar twins of every kernel, used when the Vector API module is missing, and Vector API kernels for aarch64 and for GraalVM native images.
- A GraalVM native image smoke test on Linux (x86-64 and arm64), macOS arm64 and Windows x64.
- CI on JDK 21 and 25, with the fast tests also on arm64 and on Windows x64.
- The supported public surface and the stability policy ([docs/API.md](docs/API.md)).
- A benchmark harness for the tier (`src/bench`, [docs/BENCHMARKS.md](docs/BENCHMARKS.md)) with a CI smoke at 10,000 vectors. No result file is published yet.
