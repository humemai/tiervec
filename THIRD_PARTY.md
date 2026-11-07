# Third-party code and ideas

Sources of jvector, Lucene HNSW and hnswlib (Apache-2.0 or MIT) may be read, and small pieces adapted with an attribution comment at the top of the file plus a line here. GPL or unlicensed code is never copied.

| Library or paper | Version | License | Used in | What |
|---|---|---|---|---|
| Malkov and Yashunin, HNSW (paper) | arXiv 1603.09320 | n/a (paper) | graph tier | multi-layer graph, level draw with mL = 1/ln(M), neighbor-selection heuristic, beam search per layer |
| Subramanya et al., DiskANN / Vamana (paper) | NeurIPS 2019 | n/a (paper) | graph tier | alpha robust prune; delete consolidation (repair of in-neighbors using the deleted node's out-neighbors) |
| jvector, Lucene HNSW, hnswlib | not read | Apache-2.0 / MIT | none | no code or structure copied so far; the graph tier was written from the papers. If a piece is adapted later, add a row here and an attribution comment in that file. |
