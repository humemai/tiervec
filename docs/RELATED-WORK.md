# Related work

This page compares the design of tiervec with published research and with what current systems document, for one workload: the memory of an agent inside a database engine. That means continuous creates, reads, updates, and deletes, many small and medium collections, queries that see a write at once, and filtered search. It was written on 2026-11-28 against `main` at commit `df857c3`. The benchmark that would test its judgements is defined in [AGENT-MEMORY-BENCHMARK.md](AGENT-MEMORY-BENCHMARK.md).

How to read it:

- A statement about another system comes from the linked source. [Sources](#sources) says how much of each source was read. No statement here rests on a search result summary.
- A statement marked *(inference)* is a conclusion drawn here. No source says it.
- Numbers about other systems are their authors' own measurements on their own setups. They were not reproduced.
- Each design choice gets one verdict. **Matches**: current practice does the same. **Behind**: published work or a current system does something that tiervec lacks and that matters for this workload. **Unproven**: the choice is reasonable, and no measurement shows that it holds.

## Verdict

tiervec cannot claim to be the best vector index for agent memory, and it cannot claim that its whole design follows the state of the art.

- Its overall shape is established practice: an exact buffer for new rows, immutable index segments, delete markers, background merges, and the host's rows as the source of truth.
- It is behind on filtered search, on memory per vector, on the update rules of graphs that live through deletes, and on batch visibility under a steady stream of commits.
- Nothing is measured under churn: no recall over a long stream of updates and deletes, no latency under writes against another library, and nothing about thousands of small collections.
- The largest open design question is whether segments beat one graph that is updated in place. Published work argues both ways, and no measurement here compares them.

| # | Design choice | Verdict | What settles it |
|---|---|---|---|
| 1 | [Exact tail for new rows](#freshness-the-exact-tail) | Matches | [#159](https://github.com/humemai/tiervec/issues/159), [#4](https://github.com/humemai/tiervec/issues/4) |
| 2 | [Immutable segments and background merges](#segments-or-one-graph-that-changes) | Matches, unproven against one updatable graph | [#159](https://github.com/humemai/tiervec/issues/159), [#160](https://github.com/humemai/tiervec/issues/160) |
| 3 | [Delete markers in sealed segments](#deletes-and-updates-in-graphs) | Matches, recall under churn unproven | [#159](https://github.com/humemai/tiervec/issues/159) |
| 4 | [Mutable graph and insert merge: `alpha` 1.0, no repair of low degree](#deletes-and-updates-in-graphs) | Behind | [#161](https://github.com/humemai/tiervec/issues/161), [#16](https://github.com/humemai/tiervec/issues/16) |
| 5 | [Merge by full rebuild (default)](#merging) | Behind on write cost | [#16](https://github.com/humemai/tiervec/issues/16) |
| 6 | [Merge policy and search over several segments](#merging) | Unproven | [#4](https://github.com/humemai/tiervec/issues/4) |
| 7 | [No graph below a size threshold](#small-collections-and-many-collections) | Matches, threshold unproven | [#4](https://github.com/humemai/tiervec/issues/4) |
| 8 | [One tier per collection](#small-collections-and-many-collections) | Unproven | [#165](https://github.com/humemai/tiervec/issues/165) |
| 9 | [Filtered search](#filtered-search) | Behind | [#162](https://github.com/humemai/tiervec/issues/162), [#12](https://github.com/humemai/tiervec/issues/12) |
| 10 | [Quantization and memory per vector](#quantization-and-memory-per-vector) | Behind | [#164](https://github.com/humemai/tiervec/issues/164), [#72](https://github.com/humemai/tiervec/issues/72), [#73](https://github.com/humemai/tiervec/issues/73), [#74](https://github.com/humemai/tiervec/issues/74) |
| 11 | [Hot and cold data](#hot-and-cold-data) | Behind, mostly not built | [#165](https://github.com/humemai/tiervec/issues/165), [#164](https://github.com/humemai/tiervec/issues/164) |
| 12 | [Read-your-writes](#consistency-transactions-and-concurrency) | Matches | [#159](https://github.com/humemai/tiervec/issues/159) |
| 13 | [Batch visibility by waiting and scanning again](#consistency-transactions-and-concurrency) | Behind | [#163](https://github.com/humemai/tiervec/issues/163) |
| 14 | [One write lock, many writers](#consistency-transactions-and-concurrency) | Unproven | [#60](https://github.com/humemai/tiervec/issues/60), [#159](https://github.com/humemai/tiervec/issues/159) |
| 15 | [Host rows as the truth, id epoch, replay](#persistence-and-restart) | Matches | [#78](https://github.com/humemai/tiervec/issues/78) |
| 16 | [The tail is written at a checkpoint](#persistence-and-restart) | Matches, restart time unmeasured | [#159](https://github.com/humemai/tiervec/issues/159) |

## The workload that agent memory implies

| System or benchmark | What its source says | What it asks of an index *(inference)* |
|---|---|---|
| [Mem0](https://arxiv.org/abs/2504.19413) | A language model picks `ADD`, `UPDATE`, `DELETE`, or `NOOP` for each new fact. On its benchmark the stored memory is about 7k tokens per conversation | Real updates and deletes, small collections |
| [A-MEM](https://arxiv.org/abs/2502.12110) | A new memory can trigger changes to the context and attributes of older memories | Updates of old rows, not only appends |
| [Zep and Graphiti](https://arxiv.org/abs/2501.13956) | 1024-dimension embeddings, cosine search beside full-text and graph search. A contradicted fact is marked invalid with timestamps | Time and validity filters, hybrid retrieval |
| [MemGPT](https://arxiv.org/abs/2310.08560) | Archival memory is a read and write database. Its default setup is PostgreSQL with a pgvector HNSW index | A vector index inside a database |
| [Letta archival memory](https://docs.letta.com/v1-sdk/memory/archival-memory) | A vector store for facts that the agent queries on demand and cannot easily modify or delete | Append-heavy use also exists |
| [LangGraph long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory) | JSON documents organized by namespace and key | Many small collections, or one collection with a filter |
| [Generative Agents](https://arxiv.org/abs/2304.03442) | Retrieval combines relevance, recency, and importance | The host re-ranks, so it needs exact scores and enough candidates |
| [LongMemEval](https://arxiv.org/abs/2410.10813) | About 115k tokens of history per question in the S setting, 500 sessions and about 1.5M tokens in the M setting | One user's memory is small |
| [LoCoMo](https://arxiv.org/abs/2402.17753) | Conversations of 300 turns and 9K tokens on average, up to 35 sessions | One conversation's memory is small |
| [Agent memory characterization](https://arxiv.org/abs/2606.06448) | The memory systems it measured span two orders of magnitude in query latency, with a comparable spread in construction cost | Cost sits in the whole pipeline, not only in the index |

Two conclusions follow, both *(inference)*. A conversational memory per user is likely hundreds to a few thousand vectors, so the exact path, the cost of one collection, and the number of collections matter more than the speed of one large graph. A shared team memory of 1,000,000 vectors or more ([#60](https://github.com/humemai/tiervec/issues/60)) is a second shape with different needs. Where a language model builds each memory, its calls probably cost far more than a vector search, so an index wins this workload mostly by never being wrong after a change, by using little memory, and by not disturbing the host.

## Design choices

### Freshness: the exact tail

**Sources say.** [FreshDiskANN](https://arxiv.org/abs/2105.09613) puts recent points into a small in-memory index beside a large one on SSD, with a delete list. [SingleStore-V](https://www.vldb.org/pvldb/vol17/p3772-chen.pdf) writes new rows to an in-memory rowstore that searches scan. [turbopuffer](https://turbopuffer.com/docs/architecture) searches data that is not indexed yet by an exhaustive scan of its log, and [LanceDB](https://docs.lancedb.com/enterprise/v1-14-0/indexing/incremental-indexing) combines its index with a flat search of new rows. [Milvus](https://milvus.io/docs/glossary.md) has growing segments that collect new data and sealed segments that accept none. [AnalyticDB-V](https://www.vldb.org/pvldb/vol13/p3152-wei.pdf) instead builds an HNSW index over new data in real time and merges it into the baseline later.

**tiervec today.** `add` returns after the vector is in a tail that every later search scans exactly. The tail is bounded, and a seal turns it into a graph segment ([DESIGN.md](DESIGN.md#exact-tail)).

**Verdict: matches.** The cost is a scan of the tail on every query. The right tail size, and whether a small graph on the tail would pay, are unmeasured.

### Segments or one graph that changes

**Sources say.** Segmented designs are common: FreshDiskANN merges its small index into the large one at a cost proportional to the change set, SingleStore-V keeps one immutable vector index per segment with a delete bit vector and a background merger, and Lucene and Milvus use segments. The other side is explicit too. [Azure Cosmos DB](https://arxiv.org/abs/2505.05885) keeps one DiskANN index per partition in step with the data, with in-place deletes and a background cleanup, and argues that per-segment indexes make every query visit many segments and make rebuilds cause latency spikes. [IP-DiskANN](https://arxiv.org/abs/2502.13826) processes each insert and delete in place and reports stable recall, with better query throughput and update speed than batch consolidation and than HNSW. [SPFresh](https://arxiv.org/abs/2410.14452) updates a partitioned (inverted file) index in place by splitting partitions and moving boundary vectors, and reports lower resource use than global rebuilds at billion scale with 1% daily updates. [LSM-Vec](https://arxiv.org/abs/2505.17152) and [Quake](https://arxiv.org/abs/2506.03437) are further recent designs for changing data. The [NeurIPS 2023 streaming track](https://arxiv.org/abs/2409.17424) scores indexes while a runbook inserts, deletes, and searches.

**tiervec today.** A tail, up to `maxSegments` (8 by default) immutable segments, and one atomic swap of the segment list.

**Verdict: matches common practice, unproven for this workload.** SPFresh and DiskANN target data far larger than memory, which agent memory per user is not *(inference)*. The in-place graph is a real alternative at these sizes *(inference)*, and tiervec already holds the parts for it: `GraphIndex` is mutable and concurrent and repairs itself with `consolidate()`. If a live-graph arm beats the tier on the benchmark, the design should change.

### Deletes and updates in graphs

**Sources say.** [hnswlib](https://github.com/nmslib/hnswlib) marks an element deleted, leaves it out of results, and can reuse its slot. [JVector](https://github.com/datastax/jvector) marks nodes deleted and removes them later with `removeDeletedNodes`, which its source says is not thread-safe against other modifications. FreshDiskANN shows recall falling over cycles of deletes and inserts with simple delete rules on HNSW, Vamana, and NSG graphs. With its own repair rules recall was stable for every pruning factor `alpha` it tried except 1. A [study of deletion in HNSW](https://arxiv.org/abs/2512.06200) reports that recall under delete markers alone falls as inserts and deletes repeat, and that a rebuild gives the best recall, ahead of physical deletion. [Another](https://arxiv.org/abs/2407.07871) reports that after about 3,000 rounds of delete and reinsert 3% to 4% of the points of an HNSW graph became unreachable and recall fell by about 3%. Lucene's [change log](https://github.com/apache/lucene/blob/main/lucene/CHANGES.txt) records a change (GITHUB#16552) that reconnects nodes whose out-degree decays across successive generations of reused graphs, to prevent a progressive recall loss at low delete rates.

**tiervec today.** In a sealed segment a deleted or replaced row is hidden by the version floor, still walked, and dropped at the next merge. A segment is rebuilt when 30% of it is hidden (`deadFraction`). `GraphIndex.consolidate()` repairs the lists that point at deleted nodes from the survivors around them and reconnects unreachable nodes. The default `alpha` is 1.0. `MergeMode.INSERT`, off by default, copies the largest graph, deletes from the copy, and inserts the rest. Its first check held over 10 generations at 3,000 vectors of 32 dimensions (recall from 0.0040 below to 0.0125 above a rebuild, [#16](https://github.com/humemai/tiervec/issues/16)).

**Verdict.** Delete markers with periodic rebuild: **matches**, and the version floor also covers reused record ids, which the host contract needs. The mutable graph and the insert merge: **behind**, because the two findings that apply to long-lived graphs (`alpha` above 1, and degree repair across reuse) are not in the defaults or in any test. Whether they matter at these sizes is open ([#161](https://github.com/humemai/tiervec/issues/161)).

### Merging

**Sources say.** Lucene reuses an existing graph when it merges segments and has made merges faster by joining the other graphs into the base graph (GITHUB#14331). It has also worked on the cost of searching many segment graphs (GITHUB#12962, GITHUB#13121; the method was not read). FreshDiskANN's merge costs in proportion to the change.

**tiervec today.** The default merge builds a new graph from the live vectors of its inputs, which keeps graph quality and pays for the whole result. A merge by insertion into the largest graph is an option and not the default ([study](BENCHMARKS.md#the-study-at-100000-rows)). A search visits every segment with the full beam, one segment after the other, unless `splitBeam` is on. In single runs at 100,000 vectors the default layout of 6 segments answered at 1.70 (sift128) and 2.33 (dbpedia768) times the median latency of one graph at equal recall ([#4](https://github.com/humemai/tiervec/issues/4)).

**Verdict.** Rebuild merges: **behind** on write cost. The default policy: **unproven**, with a candidate and a registered study in [#4](https://github.com/humemai/tiervec/issues/4). Sharing a search bound between segments is an idea to test there.

### Small collections and many collections

**Sources say.** Lucene builds no HNSW graph for a tiny segment (GITHUB#14963); its HNSW format has a threshold, 100 by default, compared with the number of nodes a search is expected to visit. [Weaviate](https://docs.weaviate.io/weaviate/concepts/vector-index) offers an index that starts flat and switches to HNSW above 10,000 objects by default, one way, and recommends it for many tenants. [Qdrant](https://qdrant.tech/documentation/manage-data/indexing/) answers a filtered query by a full scan when the matching points are below a size threshold. [Curator](https://arxiv.org/abs/2401.07119) states the trade-off between one index per tenant (fast, memory heavy) and one shared index with filters (lean, slow), and proposes a shared tree of per-tenant trees.

**tiervec today.** Below `smallLimit` (1,000) live vectors every search is an exact scan. A host makes one tier per collection. Without a `HostEnv` each tier starts a worker thread; with one it starts none.

**Verdict.** The threshold idea **matches**. Its value, and the whole many-collection shape, are **unproven**: no figure for the memory of a tier, for thousands of tiers, or for reopening an idle one ([#165](https://github.com/humemai/tiervec/issues/165)).

### Filtered search

**Sources say.** [ACORN](https://arxiv.org/abs/2403.04871) walks the subgraph of rows that pass the predicate. [NaviX](https://arxiv.org/abs/2506.23397), built inside a graph database, picks a rule at each step from the local share of passing neighbors. [RACORN](https://arxiv.org/abs/2607.00768) reports that ACORN-1 recall collapses below a 1% share and adds a fallback. [Filtered-DiskANN](https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf) builds label information into the graph, including a variant for streaming updates. Lucene added an ACORN-1 based search strategy (GITHUB#14160, "as much as 5x" faster filtered search by its own account). Qdrant adds graph edges per indexed payload field and offers ACORN from version 1.16.0, also naming many soft-deleted points as a cause of disconnected graphs. [pgvector](https://github.com/pgvector/pgvector) filters after the index scan and can continue the scan until enough rows pass. Inside databases, a [PostgreSQL study](https://arxiv.org/abs/2603.23710) finds that filter checks and page access can cancel the theoretical gain of graph methods, and [another study](https://arxiv.org/abs/2602.11443) finds partitioned indexes ahead of HNSW at low shares.

**tiervec today.** The host passes a predicate over ids. A set that the host says is small, or a sampled share below `filterScanBelow` (off by default), is scanned exactly. Otherwise the walk traverses every neighbor, collects those that pass, and after `max(50 * ef, rows / 20)` visits with too few results scans the whole segment. A segment with hidden rows takes this path even without a host filter. Filters correlated with the vectors, such as one user's memories, are not measured ([BENCHMARKS.md](BENCHMARKS.md#the-data)).

**Verdict: behind.** Exact fallback matches practice. A walk that adapts to the filter is missing, and the thresholds are guesses.

### Quantization and memory per vector

**Sources say.** [DiskANN](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf) serves a billion points from 64 GB of memory and an SSD, with compressed vectors in memory. JVector walks the graph on compressed codes (product, binary, and others) and reranks with full or higher-accuracy vectors. [LVQ](https://arxiv.org/abs/2304.04759) centers the data, then quantizes each vector with its own scale, and can keep a second residual for reranking. [RaBitQ](https://arxiv.org/abs/2405.12497) codes one bit per dimension with a proven error bound and reports better accuracy than product quantization; [its extension](https://arxiv.org/abs/2409.09913) allows more bits. [TurboQuant](https://arxiv.org/abs/2504.19874) needs no training on the data. Lucene has 4-bit and 8-bit optimized scalar formats and [binary formats](https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch) that use a centroid, correction terms, and reranking. Quantizers that learn from the data degrade when it drifts unless they are updated ([DeDrift](https://arxiv.org/abs/2308.02752), [streaming quantization](https://arxiv.org/abs/2512.18335)).

**tiervec today.** SQ8 with its own scale or range for each vector, which needs no training and cannot go stale with drift *(inference)*. A one-bit scan with a per-dimension threshold, frozen after the first calibration, on the flat index only; on sift128 its recall fell from 0.9859 at 100,000 to 0.8990 at 1,000,000 vectors with a rerank factor of 500 ([#73](https://github.com/humemai/tiervec/issues/73)). The fp32 vectors always stay, on the heap or in a mapped file, so SQ8 adds memory. In the posted comparison at 100,000 vectors, median peak resident memory was 1380 MB (float32) and 1483 MB (SQ8) for the tier against 1067 MB and 1090 MB for JVector float32 and int8 ([#121](https://github.com/humemai/tiervec/issues/121); the run holds the harness too, so these are not index sizes).

**Verdict: behind** on memory. The per-vector SQ8 is a sound choice under churn. The one-bit scheme is older than current one-bit methods.

### Hot and cold data

**Sources say.** DiskANN keeps compressed vectors in memory and the graph on SSD. [SPANN](https://arxiv.org/abs/2111.08566) keeps partition centroids in memory and the lists on SSD. [Milvus tiered storage](https://milvus.io/docs/tiered-storage-overview.md) loads metadata first and fetches or evicts data on demand. turbopuffer keeps namespaces on object storage. Weaviate has a partitioned index that keeps most data on disk.

**tiervec today.** The tiers are by age of the write (tail, then segments), not by use. `MAPPED` storage leaves only the fp32 vectors in the file. `open` reads and verifies a whole directory. There is no on-disk graph and no way to load part of a tier.

**Verdict: behind.** For many small collections the likely need is a cheap way to close an idle collection and reopen it, not an on-disk graph *(inference)*.

### Consistency, transactions, and concurrency

**Sources say.** Milvus defaults to [bounded staleness](https://milvus.io/docs/consistency.md). turbopuffer is strongly consistent by default. SingleStore reads a consistent snapshot with multiversion concurrency control, and [TigerVector](https://arxiv.org/abs/2501.11216) answers a query at a transaction id from index snapshots plus deltas.

**tiervec today.** A search that starts after a write returns sees it. Searches take no lock, and writers share one short write lock. A batch (`apply`) is visible as a unit because a search waits while one is open and scans again if one overlapped it; a steady stream of batches can starve a search ([DESIGN.md](DESIGN.md#batches)). There is no snapshot read at an older version: uncommitted rows are the host's overlay, and a host with snapshot isolation has to drop rows that its transaction must not see, which can leave fewer than `k` *(inference)*.

**Verdict.** Read-your-writes **matches**, and is stricter than a bounded-staleness default. Batch visibility is **behind** multiversion reads ([#163](https://github.com/humemai/tiervec/issues/163)). Many concurrent writers are **unproven**.

### Persistence and restart

**Sources say.** FreshDiskANN rebuilds its in-memory index and delete list after a crash by replaying a redo log since the last snapshot. Cosmos DB stores its index in the database's own index trees, and TigerVector reuses the database's write-ahead log and commit protocol.

**tiervec today.** The host's rows and log are the truth. Segments, tombstones, the tail, and a manifest are checkpointed and verified at `open`, and the host replays newer rows through an id epoch and a high-water id ([ID-EPOCH-AND-REPLAY.md](ID-EPOCH-AND-REPLAY.md)). `checkpoint` writes the tail as it is, without building a segment ([#158](https://github.com/humemai/tiervec/issues/158), [SEGMENT-FORMAT.md](SEGMENT-FORMAT.md#tail-file)).

**Verdict.** The host contract **matches** the practice of an index that follows the database's log, and a snapshot of the unsealed rows with a replay from the snapshot on is what FreshDiskANN describes. The whole tail is written at every checkpoint, and restart time is unmeasured.

### Hybrid search, recency, and time

Memory systems combine vector search with text search, graph steps, recency, and importance. tiervec leaves that to the host: it returns exact fp32 scores and takes the host's filter as a set of ids. That division suits an index inside a database *(inference)*, and it makes filtered search carry more weight. No verdict is given.

## Libraries and engines side by side

Only what the linked sources state. A dash means that nothing was checked.

| System | Runs inside a JVM host | New rows | Deletes | Filters |
|---|---|---|---|---|
| tiervec | Yes | Exact tail | Version floor, rebuild at merge | Predicate walk, exact fallback |
| Lucene | Yes | Segments | Delete markers, graphs reused at merge | ACORN-1 based strategy available |
| JVector | Yes | Concurrent inserts into one graph | Mark, then `removeDeletedNodes` | Allowed-node bits ([BENCHMARKS.md](BENCHMARKS.md#the-jvector-arm)) |
| hnswlib | No (C++ and Python) | Inserts into one graph | Mark deleted, replace deleted | - |
| pgvector | No (inside PostgreSQL) | - | Vacuum, which can be slow for HNSW | After the index scan, iterative scans |
| Qdrant | No (server) | - | Soft deletes | Extra graph edges, full scan, ACORN |
| Milvus | No (server) | Growing segments | - | - |
| LanceDB | - | Flat search beside the index | - | - |

## Where tiervec is not best today

- Filtered search has no filter-aware walk and untested thresholds.
- Memory per vector is bound by fp32, and was higher than JVector's in the one posted comparison.
- Restart time is unknown.
- A search can be starved by a steady stream of batches.
- The default merge rewrites whole graphs, and the default merge policy searches several segments.
- There is no evidence for any size above 1,000,000 vectors, for concurrent writers at scale, or for many collections.

What speaks for it is narrow. In one static comparison at 100,000 vectors (one data set, one search thread), the SQ8 tier searched faster than every JVector configuration at recall 0.95, with median latency ratios from 0.739 to 0.925. At recall 0.99, among the configurations that reached it, it was faster than one, the same as two, and undecided against one ([#121](https://github.com/humemai/tiervec/issues/121)). That says nothing about change over time. The repository also tests what a host needs after a change: reused ids, atomic batches, and searches that overlap a merge ([STATUS.md](STATUS.md#record-id-reuse-and-tombstones)).

## What it would take

In this order, because each step decides what the next one is worth:

1. Measure the workload: the runbook benchmark ([#159](https://github.com/humemai/tiervec/issues/159)) and its baselines, a live graph, JVector, and Lucene ([#160](https://github.com/humemai/tiervec/issues/160)).
2. Keep answers right and recall stable under change: snapshot reads for batches ([#163](https://github.com/humemai/tiervec/issues/163)), and the update rules of long-lived graphs ([#161](https://github.com/humemai/tiervec/issues/161), [#16](https://github.com/humemai/tiervec/issues/16)).
3. Filtered search ([#162](https://github.com/humemai/tiervec/issues/162), [#12](https://github.com/humemai/tiervec/issues/12)).
4. Many collections, idle ones included, and restart ([#165](https://github.com/humemai/tiervec/issues/165), [#158](https://github.com/humemai/tiervec/issues/158)).
5. Memory per vector ([#164](https://github.com/humemai/tiervec/issues/164), [#72](https://github.com/humemai/tiervec/issues/72), [#73](https://github.com/humemai/tiervec/issues/73), [#74](https://github.com/humemai/tiervec/issues/74)).
6. Merge cost and layout ([#4](https://github.com/humemai/tiervec/issues/4), [#16](https://github.com/humemai/tiervec/issues/16)).

The API is not stable yet, so a result that favors another design should change the design, breaking changes included.

## Sources

"Passages" means that the abstract was read and the full text was searched for the points used here. It does not mean that the whole paper was read. Documentation and change logs were fetched on 2026-11-28 and searched the same way.

| Source | Read |
|---|---|
| FreshDiskANN [2105.09613](https://arxiv.org/abs/2105.09613), SPFresh [2410.14452](https://arxiv.org/abs/2410.14452), NeurIPS 2023 results [2409.17424](https://arxiv.org/abs/2409.17424), HNSW deletion study [2512.06200](https://arxiv.org/abs/2512.06200), unreachable points [2407.07871](https://arxiv.org/abs/2407.07871) | Passages |
| [Cosmos DB](https://arxiv.org/abs/2505.05885), [SingleStore-V](https://www.vldb.org/pvldb/vol17/p3772-chen.pdf), [TigerVector](https://arxiv.org/abs/2501.11216), [AnalyticDB-V](https://www.vldb.org/pvldb/vol13/p3152-wei.pdf), [DiskANN](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf) | Passages |
| ACORN [2403.04871](https://arxiv.org/abs/2403.04871), NaviX [2506.23397](https://arxiv.org/abs/2506.23397), RACORN [2607.00768](https://arxiv.org/abs/2607.00768), PostgreSQL filter study [2603.23710](https://arxiv.org/abs/2603.23710), LVQ [2304.04759](https://arxiv.org/abs/2304.04759), RaBitQ [2405.12497](https://arxiv.org/abs/2405.12497), streaming quantization [2512.18335](https://arxiv.org/abs/2512.18335) | Passages |
| Mem0 [2504.19413](https://arxiv.org/abs/2504.19413), Zep [2501.13956](https://arxiv.org/abs/2501.13956), MemGPT [2310.08560](https://arxiv.org/abs/2310.08560), A-MEM [2502.12110](https://arxiv.org/abs/2502.12110), Generative Agents [2304.03442](https://arxiv.org/abs/2304.03442), LongMemEval [2410.10813](https://arxiv.org/abs/2410.10813), agent memory characterization [2606.06448](https://arxiv.org/abs/2606.06448) | Passages |
| IP-DiskANN [2502.13826](https://arxiv.org/abs/2502.13826), SPANN [2111.08566](https://arxiv.org/abs/2111.08566), LSM-Vec [2505.17152](https://arxiv.org/abs/2505.17152), Quake [2506.03437](https://arxiv.org/abs/2506.03437), Curator [2401.07119](https://arxiv.org/abs/2401.07119), [Filtered-DiskANN](https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf), filtered search design study [2602.11443](https://arxiv.org/abs/2602.11443), extended RaBitQ [2409.09913](https://arxiv.org/abs/2409.09913), TurboQuant [2504.19874](https://arxiv.org/abs/2504.19874), DeDrift [2308.02752](https://arxiv.org/abs/2308.02752), LoCoMo [2402.17753](https://arxiv.org/abs/2402.17753) | Abstract |
| [Lucene change log](https://github.com/apache/lucene/blob/main/lucene/CHANGES.txt) and [HNSW format source](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/codecs/lucene99/Lucene99HnswVectorsFormat.java), [JVector README](https://github.com/datastax/jvector) and graph builder source, [hnswlib README](https://github.com/nmslib/hnswlib), [pgvector README](https://github.com/pgvector/pgvector) | Documentation and source, searched |
| Qdrant [indexing](https://qdrant.tech/documentation/manage-data/indexing/) and [quantization](https://qdrant.tech/documentation/manage-data/quantization/), Milvus [glossary](https://milvus.io/docs/glossary.md), [consistency](https://milvus.io/docs/consistency.md), and [tiered storage](https://milvus.io/docs/tiered-storage-overview.md), [Weaviate](https://docs.weaviate.io/weaviate/concepts/vector-index), [turbopuffer](https://turbopuffer.com/docs/architecture), [LanceDB](https://docs.lancedb.com/enterprise/v1-14-0/indexing/incremental-indexing), [Elastic on binary quantization](https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch), [Letta](https://docs.letta.com/v1-sdk/memory/archival-memory), [LangGraph](https://docs.langchain.com/oss/python/langchain/long-term-memory) | Documentation, searched |

Not covered: GPU indexes (a [non-goal](ROADMAP.md#non-goals)), sparse vectors, learned indexes, and the many filtered-search methods for numeric ranges. An earlier internal survey compared libraries and systems for an engine integration. What is reused from it here was checked against the sources above.
