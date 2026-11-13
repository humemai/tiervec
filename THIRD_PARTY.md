# Third-party code and ideas

Sources of jvector, Lucene HNSW and hnswlib (Apache-2.0 or MIT) may be read, and small pieces adapted with an attribution comment at the top of the file plus a line here. GPL or unlicensed code is never copied.

| Library or paper | Version | License | Used in | What |
|---|---|---|---|---|
| Malkov and Yashunin, HNSW (paper) | arXiv 1603.09320 | n/a (paper) | graph tier | multi-layer graph, level draw with mL = 1/ln(M), neighbor-selection heuristic, beam search per layer |
| Subramanya et al., DiskANN / Vamana (paper) | NeurIPS 2019 | n/a (paper) | graph tier | alpha robust prune; delete consolidation (repair of in-neighbors using the deleted node's out-neighbors) |
| jvector, Lucene HNSW, hnswlib | not read for the graph tier | Apache-2.0 / MIT | none | no code or structure copied so far; the graph tier was written from the papers. If a piece is adapted later, add a row here and an attribution comment in that file. |
| jvector (`io.github.jbellis:jvector`) | 4.0.2 | Apache-2.0 | the benchmark harness only (`src/bench/jvector`, profile `bench` of `pom.xml`) | the comparison arm of [docs/BENCHMARKS.md](docs/BENCHMARKS.md): the harness builds and searches a JVector graph through its public API. It is a dependency of the harness, not of the library, and no JVector code is copied. |
| Sivic and Zisserman, inverted file (paper) | ICCV 2003 | n/a (paper) | IVF index | inverted file over k-means centroids |
| Jegou, Douze and Schmid, product quantization, IVFADC (paper) | TPAMI 2011 | n/a (paper) | IVF index | the IVF part only; no product quantization or residual codes |
| Arthur and Vassilvitskii, k-means++ (paper) | SODA 2007 | n/a (paper) | IVF index | seeding of the Lloyd k-means iterations; no third-party source was read or adapted |
