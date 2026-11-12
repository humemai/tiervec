# Design

This page describes what the code in `src/main/java/ai/humem/tiervec` does today. It is written from the sources and their Javadoc. Where something is not built yet, it says so.

## The contract: `VecIndex`

Every index implements `VecIndex`, over `float[]` vectors with a `long` id and one of three metrics (`COSINE`, `DOT`, `L2`).

- `add`, `delete` and `search` may be called concurrently from many threads, and a search never throws because of a concurrent write.
- Read-your-writes: after `add(id, v)` returns, a search that starts later in any thread can return `id`. After `delete(id)` returns, a search that starts later never returns `id`.
- Ids are unique: adding an existing id replaces its vector.
- `Hit.score` is a similarity and higher is better for every metric (cosine and dot as they are, L2 as the negative squared distance).
- `search(q, k, allow)` takes an optional `LongPredicate` filter and may return fewer than `k` hits only when fewer than `k` live vectors qualify.

## The parts

| Class | What it is |
|---|---|
| `FlatIndex` | Exact or quantized scan over every live vector. No graph. Has `serialize` and `load`. |
| `GraphIndex` | Mutable, concurrent HNSW-style graph. `seal()` turns it into a `SealedGraph`, `fromSealed` turns it back. |
| `SealedGraph` | Immutable compact graph. The only mutation is a soft delete through a bitset. Has `serialize` and `load` (one segment file, see [SEGMENT-FORMAT.md](SEGMENT-FORMAT.md)). |
| `IvfIndex` | Standalone inverted-file index with k-means lists and a background retrain. Not a segment type of `TieredIndex`. |
| `TieredIndex` | The tier: an exact tail in front of sealed graph segments. A settled tier can be written with `checkpoint` and read with `open`. |
| `Kernels`, `VKernels`, `NKernels`, `Kern`, `VKern` | Distance kernels. `Kernels` picks vector or scalar code once per JVM or image, for the flat, IVF, and graph kernels. `NKernels` holds the Vector API kernels for a native image: the fp32 dot and squared L2, the int8 dot, the SQ8 dots, and block16; its 128-bit forms of the last four are also what an aarch64 JVM uses for them. |

### Flat index

Vectors live in chunks (the first chunks grow from 64 to 4096 slots, then 4096 per chunk), each with a float array, an id array, a live bitmap, and optional quantized codes. A slot is written once and never reused, so a reader never sees a half-written slot. Appends never copy old data.

The read path takes no lock. A writer fills a slot and then sets its live bit with release semantics, which publishes it. Delete clears the bit. Replace writes a new slot, sets it live, and clears the old bit, so a search can briefly see both copies and the result is de-duplicated by id. Writers take one of 64 stripe monitors by id hash. `compact()` takes an exclusive lock that pauses writers, never searches.

Quantization is chosen per index:

- `NONE`: fp32 scan. The result equals brute force, with ties broken by the smaller id.
- `SQ8`: symmetric int8 with one scale per vector. No training, so churn cannot make it stale.
- `BINARY`: one sign bit per dimension against a per-dimension threshold, frozen after the first calibration set of live vectors. Until then searches are exact.

A quantized search scans the codes, keeps the best `rerankFactor * k` candidates, rescores them in fp32, and returns the best `k`. Returned scores are always exact fp32 scores.

### Graph index

A multi-layer HNSW graph, with the alpha robust prune of DiskANN (Vamana) as a parameter (`alpha = 1` is the HNSW neighbor heuristic). Nodes live in fixed-size chunks and never move. Neighbor lists are copy-on-write arrays published with release and acquire `VarHandle` stores, so readers take no lock and always see a complete list. Writers take a striped monitor of the owning node, one at a time.

- `add` publishes the node's own links, then marks it live, then adds the reverse links, then updates the entry point. A node is returnable only when live, and reachable only through published links.
- `delete` marks a node deleted at once. Deleted nodes are still traversed. `consolidate()` repairs the neighbor lists that hold a tombstone, waits (never blocking a reader) until searches that started earlier are done, and then recycles the slots.
- A replace is a new node plus a tombstone of the old one.
- Search is a beam search at layer 0 with a greedy descent above. Filtered-out and deleted nodes are traversed but not collected. With a filter, after a visit budget with fewer than `k` collected, the search falls back to an exact scan over the nodes that pass the filter.
- `filterScanBelow(fraction)` (default 0, off) lets a filtered search skip the walk when the allowed share is small. The share is estimated from 256 id slots spread over the index (a fixed number of predicate calls, no size hint from the host). When at most `fraction` of the sampled live slots pass, the search scans the allowed ids exactly, which costs one predicate call per live id and returns the exact top `k`. Above the threshold the search is unchanged. With the option off, a filtered search spends its whole visit budget before the scan when the allowed share is below about `ef` divided by the budget, and that wasted walk is most of the cost of a filtered search at a share near 0.01 (see issues 12 and 56).
- With `Params.sq8` the traversal uses int8 codes and the final candidates are rescored in fp32. Construction always uses fp32.

### IVF index

K-means centroids, one append-only list per centroid, and a search probes the nearest `nprobe` lists. Until the live count reaches `trainAt`, every search is an exact scan. Training is lazy. `needsRetrain()` measures drift, list skew, and growth, and `retrain()` rebuilds the model off to the side while writers continue (their changes go to a log that is replayed), then swaps the new model in with one volatile write. Searches never wait for a retrain. Deletes are tombstones in the lists, and `compact()` rewrites a list from its live slots.

IVF has no serialization yet, and it is not used by `TieredIndex`.

## The tier

`TieredIndex` is the composition. The vectors in the host's records are the source of truth. The tail, the segments, and the floor map are derived data that can be thrown away and rebuilt (`rebuildFrom`) from raw vectors, so the code never has to make the derived state durable. `rebuildFrom` swaps the rebuilt state in the way a merge does: a search that started before the swap keeps the floor entries that its segments need, and they are dropped when that search is done.

### Exact tail

`add` puts the vector into the active tail, a `FlatIndex` (fp32 or SQ8), and a raw copy next to it. The raw copy is what a seal and a rebuild read. The vector is visible to the next search in any thread before `add` returns. `delete` removes it from the tails and hides every segment copy at once. `apply` does both for a whole batch (see Batches).

### Sealed HNSW segments

When the active tail reaches `tailLimit` vectors, or its first vector is older than `tailAgeMs`, and the index holds at least `smallLimit` vectors, one worker thread seals it:

1. Freeze the tail under the write lock. A new empty tail takes over writes.
2. Build a `GraphIndex` from the frozen contents and seal it, outside any lock, while searches and writes continue (searches scan both tails).
3. Publish the segment with one volatile write of an immutable view that drops the frozen tail.

A vector is always in the frozen tail, in the segment, or in both for an instant, and the dedupe resolves the overlap, so it is never missing. Below `smallLimit` live vectors no graph is built at all and every search is an exact scan.

### Atomic swap of the segment list

The segment list, the active tail, and the frozen tail live in one immutable view. A seal or a merge builds its result off to the side and publishes a new view with a single volatile write. A search reads the view once and finishes on it, so a search that started before a swap still sees everything it needs and the old segments stay valid for it. To keep that true for hidden copies, each view counts the searches running on it: a search raises the count, reads the view again to check that it is still current, and lowers the count when it is done (one atomic add each way, no lock). A view that a swap replaces while its count is not zero is held until the count reaches zero.

### Batches

`apply(batch)` is the write path of a host transaction: the adds, then the deletes, under one hold of the write lock. The tail and the floor map change in place, so a new view cannot make a batch visible at one instant. A batch counter does: it is odd while a batch is applied and even otherwise, and only the writer that holds the lock changes it. A search reads it before it scans and waits while it is odd. After the scan it reads it again, behind an acquire fence. When the value changed, a batch was applied during the scan, the answer may mix the state before the batch with the state after it, and the search runs again on the current view. A search that returns therefore saw no batch in progress from the start of its scan to its end: all of every earlier batch and nothing of any later one. A single `add` or `delete` does not touch the counter, because each is atomic by itself.

A search that is in the middle of a scan when a batch opens finishes the scan on the view it holds, which is still valid memory, and discards the answer. A search that starts inside the batch waits for it. A search that starts after the batch sees all of it, because the closing write of the counter comes after every change of the batch.

The query path pays one volatile read before the scan and one fence and one volatile read after it, plus a second scan when a batch overlapped the first. A search that meets a batch waits for the length of the batch (a short spin, then parks of 20 microseconds). A stream of batches that follow each other faster than one scan takes can starve a search, so the rate of `apply` calls is the bound.

### Tombstones and the version floor

Versions come from one counter under the write lock. The floor map says "a segment copy of `id` is valid only when its version is at least `floor[id]`". Replace and delete raise the floor, but only when a segment or the frozen tail holds a copy of the id, so the map stays small. Searches filter segments through the floor inside the beam, so a deleted or replaced vector never appears and nothing waits for a merge. Each segment counts its hidden copies, and a segment with none is searched with the plain filter. Because the floor is a version and not a flag, an id that the host reuses after a delete is added with a newer version, so the new vector is visible while the old copies stay hidden.

### Search

A search scans the tails exactly, searches every segment, drops hidden ids, applies the filter in both, and merges by score. Duplicate ids (a replace in flight) are resolved newest source first: active tail, frozen tail, then segments. `Freshness.SEALED_ONLY` skips the tails. It may lag (a vector added or replaced since the last seal is not in it) but never returns a deleted id, a stale vector, or an id that was never added.

### Merge

When the segment count exceeds `maxSegments`, or one segment is `deadFraction` hidden, the worker rebuilds the smallest segments (or the dead one) into one segment from their valid vectors only, throttled to `mergeRatePerSec` inserts per second so the cores that serve queries are not starved, and swaps it in. Floor entries whose stale copies are gone are dropped at the swap, except those that a segment of a held view still needs: a search that started before the swap can still be scanning an old segment, and without the entry its hidden copy would be valid again. The rest is dropped when the searches of the held views are done: the worker looks again every 5 ms while such a cleanup is pending, and `awaitIdle`, `mergeNow`, `sealNow` and `checkpoint` finish it on the calling thread, so a tier that `awaitIdle` reports idle has a settled floor map. The optional size-ratio trigger (`mergeTriggerRatio`) is described with the knobs.

### Bounds and threads

One worker thread runs one task at a time (a seal or a merge). There is a cap on segments. No allocation proportional to `n` happens on the query path. Back pressure: when the active tail holds `hardTailFactor * max(tailLimit, smallLimit)` vectors, `add` waits for the worker, bounded to 10 s. `awaitCapacity(deadline)` is the same wait as a pre-check for `apply`: it waits for room in the tail and returns `false` at the deadline. `apply` never waits for the worker and does not look at the limit, so one batch can take the tail above it. `add` and `delete` take one write lock for a short time, and `apply` takes it once for the whole batch. Searches take no lock.

## Kernels and the scalar twin

`Kernels` holds the flat-tier distance kernels: float dot and L2, int8 dot, and Hamming. Each has a Vector API version in `VKernels` and a scalar twin. The choice is made once, when `Kernels` is initialized: scalar when `-Dai.humem.tiervec.scalar=true`, scalar when `VKernels` cannot load (for example without the `jdk.incubator.vector` module), vector otherwise. `Kernels.VECTOR` says which one is in use. `IvfIndex` uses the same kernels.

On a JVM whose preferred float species has fewer than 8 lanes (128 bits: NEON on aarch64), `VKernels` has no fast form of the int8 dot, the SQ8 dots, and block16: its int8 dot is built on 256-bit species, which the JVM cannot map to NEON (138 times slower than the scalar twin in CI, [#52](https://github.com/humemai/tiervec/issues/52)), and the other three need 8 float lanes. There `Kernels` uses the 128-bit forms of `NKernels` on aarch64 (`Kernels.JVM_NARROW`) and the scalar twins on any other host (`Kernels.JVM_QUANT_SCALAR`), and `Kernels.quantReason` says which and why. If `NKernels` cannot initialize on an aarch64 JVM, the scalar twins are used, never an exception. The fp32 dot and squared L2 stay in `VKernels` on every JVM, hamming stays scalar, and a JVM with 8 or more float lanes is unchanged. `Kern.dotQ8` follows the same rule, so one run uses one SQ8 dot in the flat and in the graph tier. AVX-512 and SVE are not used.

Inside a GraalVM native image (`Kernels.IN_IMAGE`) the choice is different. `VKernels` is more than 10 times slower than scalar code there (a masked tail and a slice-based sum send the whole method through a slow path), so an image uses `NKernels` for every kernel that has a Vector API form (the fp32 dot and squared L2, the int8 dot, the SQ8 dots, and block16) when the module is present in the image, and the scalar twins otherwise. `NKernels` is the textbook form that an image compiles to SIMD code: a `static final` species, two accumulators, `fma`, `reduceLanes(ADD)`, a scalar tail, no masks. The species is chosen from `os.arch` in a static initializer, 128 bits on aarch64 and 256 bits everywhere else (a CPU with AVX-512 runs the 256-bit species), and the image builder must run that initializer, which `META-INF/native-image/ai.humem/tiervec/native-image.properties` asks for with `--initialize-at-build-time=ai.humem.tiervec.NKernels`. The initializer catches a missing module and leaves the species null, so an image built without the module still builds and `Kernels` selects the scalar twins, with the cause in `Kernels.scalarReason`. The int8, SQ8, and block16 kernels have two forms each in `NKernels`, one per species, chosen by `NKernels.WIDE`: the wide form for the 256-bit float species, and the narrow form for the 128-bit one, where the Vector API has no 32-bit byte species, so the byte vector is loaded at 64 or 128 bits and split into parts. Each form is the one that was fastest in an image on both Oracle GraalVM 25 and GraalVM CE 25.0.2, on x86-64 and arm64 Linux ([#48](https://github.com/humemai/tiervec/issues/48)). Hamming has no Vector API form and stays scalar. `Kernels.JVM_VECTOR` and `Kernels.NATIVE_VECTOR` say which holder is in use. The fixed-order sum of the JVM kernels is not used in an image: a sum in a fixed order is fast only on Oracle GraalVM, so `NKernels` takes `reduceLanes`, and its result can differ from the JVM by rounding. To build the image, give the builder `--add-modules jdk.incubator.vector -H:+VectorAPISupport` (GraalVM 25; the properties file does not set them, so a scalar image stays possible).

`Kern` holds the graph kernels: float dot, squared L2, and the dot of a float query with SQ8 codes (in an image, `NKernels.dotF8`). Its Vector API versions are in `VKern`, and its scalar twins are the scalar twins of `Kernels`. `Kern` follows `Kernels.VECTOR`, and `Kernels` initializes both holder classes inside its `try`, so one run never mixes the two modes and a failure to load either holder selects the scalar twins for every index. `GraphIndex`, `SealedGraph`, and `TieredIndex` need no incubator class in that case.

A scalar result and a vector result of the same float kernel can differ by rounding. `KernDiffTest` states the tolerance (1e-5 times the sum of the absolute terms) and compares exactly representable data bit for bit; for `NKernels` it uses 1e-4 times the sum of the absolute terms plus a floor of 1e-6, on lengths around the lane width and at 100, 128, and 770, for every form of every `NKernels` kernel, so the 128-bit forms of aarch64 are tested on an x86 host too. `NoIncubatorTest` starts a JVM without the module and builds and searches a tier in it. The graph, segment, and tier tests also run with the scalar twins, in the surefire execution `scalar-twin`.

## Tier configuration

`TieredIndex.Config` is a record. `Config.defaults()` gives the defaults below, and `with...` methods return a changed copy. The constructor rejects `tailLimit < 1`, `smallLimit < 0`, `maxSegments < 1`, `mergeFan < 2`, `hardTailFactor < 2`, `buildThreads < 1`, a negative or infinite `mergeTriggerRatio`, and a `BINARY` tail quantizer.

| Knob | Default | What it does |
|---|---|---|
| `tailLimit` | 2000 | Seal when the active tail has this many vectors. |
| `tailAgeMs` | 60000 | Also seal when the oldest tail vector is older than this. 0 means never by age. |
| `smallLimit` | 1000 | Below this many live vectors no graph is ever built and every search is an exact scan. |
| `maxSegments` | 8 | Merge when the segment count exceeds this. |
| `mergeFan` | 4 | How many of the smallest segments one merge takes. |
| `deadFraction` | 0.3 | Compact (merge) a segment when this share of it is hidden. |
| `mergeRatePerSec` | 50000 | Insert rate cap of the merge builder. 0 means unlimited. |
| `hardTailFactor` | 4 | Back pressure: writers wait when the active tail holds this factor times `max(tailLimit, smallLimit)` vectors. |
| `tailQuant` | `NONE` | Tail kernel: `NONE` scans fp32, `SQ8` scans int8 codes and reranks the best 5 times `k` candidates in fp32. |
| `graph` | `GraphIndex.Params.defaults()` | Segment graph parameters: `M` 16, `efConstruction` 200, `efSearch` 100 (the search beam), `alpha` 1.0, `sq8` false, `seed` 42. |
| `buildThreads` | `max(1, min(4, cpus - 1))` | Threads of one seal or merge build. |
| `splitBeam` | false | When true and more than one segment is searched, segment `s` gets the beam `max(k, ceil(ef * rows(s) / rows of all searched segments))` instead of the full `ef`. |
| `mergeTriggerRatio` | 0.0 | Size-ratio merge trigger, off at 0. See below. |

With `mergeTriggerRatio` X above 0, the segment with the most live rows is the big one. When the live rows of all the other segments together exceed X times the live rows of the big one, all segments are merged into one. Otherwise, if `maxSegments` is at least 2 and the count exceeds it, the count rule picks among the small segments only, so the big one is rewritten only by the ratio rule. With `maxSegments` 1 the count rule includes the big one and a merge happens at every seal. A merge into the big segment rewrites about (1 + X) / X vectors per inserted vector, on top of the small-segment merges.

Other controls on the tier: `filterScanBelow(fraction)` (default 0, off) is the scan-first threshold of filtered searches, applied per segment (the tails are always scanned exactly), `sealNow()`, `mergeNow()` and `awaitIdle(timeoutMs)` run or wait for the worker, `search(q, k, allow, ef)` sets the beam per call, and `stats()` reports seals, merges, segment sizes, floor entries, write amplification, and a log of finished tasks.
