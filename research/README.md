# Research records

These are records of studies I ran for tiervec: plans written before the runs, the scripts that ran them, and result files. They are not supported code. They are not part of the library, the build does not compile them, and some of them do not run as they stand.

| Folder | Issue | What is in it |
|---|---|---|
| [`106-real-embeddings/`](106-real-embeddings) | [#106](https://github.com/humemai/tiervec/issues/106) | The plan, the run script, the data preparation script, and the reader script of the recall study on real embeddings. The result files are in [`results/106/real/`](../results/106/real). |
| [`117-jvector-int8/`](117-jvector-int8) | [#117](https://github.com/humemai/tiervec/issues/117) | The plan, the benchmark harness, the scripts, and the result files of jvector INT8 against FP32 on jvector 4.1.0-rc.1, with the native and the Panama vectorization provider. |

To print the tables of #106 from the published result files, run this from the repository root:

```
uv run --no-project python -I research/106-real-embeddings/read_results.py --results results/106/real
```

## Edits

I replaced host names and local paths with neutral ones. In the result files of #117, `<study-dir>` stands for the study folder and `benchmark-host` for the machine, and the host environment file is called `env_host.sh`. Both plans are edited copies. The checksums that `env.txt` lists as kit sums are those of the files as they ran, so they do not match the files that I edited.

## What is not here

- Datasets, jars, compiled classes, and graph files.
- Raw JMH output and run logs of #117, because they carry local paths. The `fig-*.jsonl` files hold the per-round values that the intervals were computed from.
- The interval script (`ab.py`) that `to_jsonl_java.py` calls. Each interval is the median of the per-round ratios (INT8 time over FP32 time, or FP32 throughput over INT8 throughput) with a bootstrap 95% interval (10,000 resamples, seed 1). Round 0 is dropped as warm-up, and a pair is dropped when the two runs differ too much in thermal throttling or start temperature.
- The host environment file, the checksum lists (`SHA256SUMS`) that both run scripts check before they start, and the Lucene arm (`LuArm.java`), which the `pom.xml` of this study does not compile.
