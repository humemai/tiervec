# Roadmap

The order of work follows one question: what does a database that embeds this index need from it? The list is ordered by that need. It has no dates. Each item is a GitHub issue with a problem statement, an acceptance test, and an explicit out-of-scope list. What exists today is in [STATUS.md](STATUS.md), and the architecture is in [DESIGN.md](DESIGN.md).

## What a host needs

A host is the database that runs the index in its own process. These are the properties that decide whether an index fits a host, as opposed to fitting a benchmark:

- A committed vector is found by the next query, exactly, also across a seal, a merge, or a swap.
- Update and delete stay correct when the host reuses a record id.
- A batch of adds and deletes becomes visible as a unit.
- The index survives a crash and reopens without a full rebuild, and a damaged file is detected, never trusted.
- A filter from the host's query language (an allowed set of ids) is fast and correct at every selectivity.
- Memory is bounded, and the host, not the index, owns threads, build concurrency, and the memory budget.
- Rebuild and merge cost does not stall queries while writes continue.
- Small and medium collections work well by default, with few knobs.
- It runs on the JVM and in a native image.

## Now

Decisions and tests that the rest of the work depends on.

- [#4](https://github.com/humemai/tiervec/issues/4): Choose the default merge policy: one large segment plus a small recent one
- [#5](https://github.com/humemai/tiervec/issues/5): Tests for record-id reuse and tombstone correctness across seal and merge
- [#6](https://github.com/humemai/tiervec/issues/6): Specify the segment file and the manifest
- [#7](https://github.com/humemai/tiervec/issues/7): Define the id epoch and the replay protocol for a host. Page: [ID-EPOCH-AND-REPLAY.md](ID-EPOCH-AND-REPLAY.md).
- [#8](https://github.com/humemai/tiervec/issues/8): Document the host contract in docs/PROVIDER-CONTRACT.md. Page: [PROVIDER-CONTRACT.md](PROVIDER-CONTRACT.md).

## Next

Library work that follows from the items above. CI already runs the fast and the slow tests on JDK 21 and 25, so there is no separate item for that.

- [#9](https://github.com/humemai/tiervec/issues/9): Serialize sealed segments with a manifest, and reopen without a full rebuild. First cut: [SEGMENT-FORMAT.md](SEGMENT-FORMAT.md). Memory-mapped readers, non-empty tombstones, tail replay, and lazy verification are not in it.
- [#10](https://github.com/humemai/tiervec/issues/10): Atomic apply(batch): adds and deletes become visible together
- [#11](https://github.com/humemai/tiervec/issues/11): Search context with a deadline and cancellation
- [#12](https://github.com/humemai/tiervec/issues/12): Allowed-id filters: measure the selectivity range and add fast paths
- [#13](https://github.com/humemai/tiervec/issues/13): Host-owned executor, build permits, and memory budget instead of own threads
- [#18](https://github.com/humemai/tiervec/issues/18): Configuration map, fingerprint, capability bits, and contract version
- [#14](https://github.com/humemai/tiervec/issues/14): Scalar twin of the graph kernel. `Kern` follows the same once-per-JVM switch as the flat kernels, so the graph and the tier also run without the Vector API module.
- [#15](https://github.com/humemai/tiervec/issues/15): Native-image build and smoke test in CI
- [#16](https://github.com/humemai/tiervec/issues/16): Merge by insertion into the largest segment
- [#17](https://github.com/humemai/tiervec/issues/17): Crash and corruption tests for the segment format

## Later

One tracking issue with a checklist: [#19](https://github.com/humemai/tiervec/issues/19): Later: work that waits until the host contract holds

It covers 1-bit and 4-bit quantization schemes, IVF as a segment format, an on-disk graph, and other language bindings. Each item becomes its own issue when it is promoted.

## How to read measurements

Measurements will be published with the harness, and none are published yet. A single run (n=1) is descriptive: it shows what happened, not what usually happens. A claim of a 1.3x difference or larger needs at least five runs, each in a fresh JVM, and a smaller difference needs more. Every study states its claim, its arms, its number of runs, and the bound it must meet before it runs. Memory figures say which unit and which counter they use, and latency figures say whether the JIT was warm.

## Non-goals

- A server. The index is a library that runs in the host's process, with no network between the host and the index.
- A query language. The host parses SQL, Cypher, or anything else, and hands the index vectors and allowed-id sets.
- Its own persistence of records. The host's records are the source of truth, and everything the index stores is derived and can be rebuilt from them.
- GPU index builds or search.
