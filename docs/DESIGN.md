# Design

This page describes what the code in `src/main/java/com/humemai/vec` does today. It is written from the sources and their Javadoc. Where something is not built yet, it says so.

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
| `Kernels`, `VKernels`, `Kern` | Distance kernels. `Kernels` picks vector or scalar code once per JVM. |

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
- With `Params.sq8` the traversal uses int8 codes and the final candidates are rescored in fp32. Construction always uses fp32.

### IVF index

K-means centroids, one append-only list per centroid, and a search probes the nearest `nprobe` lists. Until the live count reaches `trainAt`, every search is an exact scan. Training is lazy. `needsRetrain()` measures drift, list skew, and growth, and `retrain()` rebuilds the model off to the side while writers continue (their changes go to a log that is replayed), then swaps the new model in with one volatile write. Searches never wait for a retrain. Deletes are tombstones in the lists, and `compact()` rewrites a list from its live slots.

IVF has no serialization yet, and it is not used by `TieredIndex`.

## The tier

`TieredIndex` is the composition. The vectors in the host's records are the source of truth. The tail, the segments, and the floor map are derived data that can be thrown away and rebuilt (`rebuildFrom`) from raw vectors, so the code never has to make the derived state durable.

### Exact tail

`add` puts the vector into the active tail, a `FlatIndex` (fp32 or SQ8), and a raw copy next to it. The raw copy is what a seal and a rebuild read. The vector is visible to the next search in any thread before `add` returns. `delete` removes it from the tails and hides every segment copy at once.

### Sealed HNSW segments

When the active tail reaches `tailLimit` vectors, or its first vector is older than `tailAgeMs`, and the index holds at least `smallLimit` vectors, one worker thread seals it:

1. Freeze the tail under the write lock. A new empty tail takes over writes.
2. Build a `GraphIndex` from the frozen contents and seal it, outside any lock, while searches and writes continue (searches scan both tails).
3. Publish the segment with one volatile write of an immutable view that drops the frozen tail.

A vector is always in the frozen tail, in the segment, or in both for an instant, and the dedupe resolves the overlap, so it is never missing. Below `smallLimit` live vectors no graph is built at all and every search is an exact scan.

### Atomic swap of the segment list

The segment list, the active tail, and the frozen tail live in one immutable view. A seal or a merge builds its result off to the side and publishes a new view with a single volatile write. A search reads the view once and finishes on it, so a search that started before a swap still sees everything it needs and the old segments stay valid for it.

### Tombstones and the version floor

Versions come from one counter under the write lock. The floor map says "a segment copy of `id` is valid only when its version is at least `floor[id]`". Replace and delete raise the floor, but only when a segment or the frozen tail holds a copy of the id, so the map stays small. Searches filter segments through the floor inside the beam, so a deleted or replaced vector never appears and nothing waits for a merge. Each segment counts its hidden copies, and a segment with none is searched with the plain filter. Because the floor is a version and not a flag, an id that the host reuses after a delete is added with a newer version, so the new vector is visible while the old copies stay hidden.

### Search

A search scans the tails exactly, searches every segment, drops hidden ids, applies the filter in both, and merges by score. Duplicate ids (a replace in flight) are resolved newest source first: active tail, frozen tail, then segments. `Freshness.SEALED_ONLY` skips the tails. It may lag (a vector added or replaced since the last seal is not in it) but never returns a deleted id, a stale vector, or an id that was never added.

### Merge

When the segment count exceeds `maxSegments`, or one segment is `deadFraction` hidden, the worker rebuilds the smallest segments (or the dead one) into one segment from their valid vectors only, throttled to `mergeRatePerSec` inserts per second so the cores that serve queries are not starved, and swaps it in. Floor entries whose stale copies are gone are dropped at the swap. The optional size-ratio trigger (`mergeTriggerRatio`) is described with the knobs.

### Bounds and threads

One worker thread runs one task at a time (a seal or a merge). There is a cap on segments. No allocation proportional to `n` happens on the query path. Back pressure: when the active tail holds `hardTailFactor * max(tailLimit, smallLimit)` vectors, writers wait for the worker, bounded to 10 s. `add` and `delete` take one write lock for a short time. Searches take no lock.

## Kernels and the scalar twin

`Kernels` holds the flat-tier distance kernels: float dot and L2, int8 dot, and Hamming. Each has a Vector API version in `VKernels` and a scalar twin. The choice is made once, when `Kernels` is initialized: scalar when `-Dhumemai.vec.scalar=true`, scalar inside a GraalVM native image, scalar when `VKernels` cannot load (for example without the `jdk.incubator.vector` module), vector otherwise. `Kernels.VECTOR` says which one is in use. `IvfIndex` uses the same kernels.

The graph kernels in `Kern` use the Vector API directly and have no scalar twin yet, so `GraphIndex`, `SealedGraph`, and `TieredIndex` need the `jdk.incubator.vector` module at run time.

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

Other controls on the tier: `sealNow()`, `mergeNow()` and `awaitIdle(timeoutMs)` run or wait for the worker, `search(q, k, allow, ef)` sets the beam per call, and `stats()` reports seals, merges, segment sizes, floor entries, write amplification, and a log of finished tasks.
