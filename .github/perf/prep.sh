#!/usr/bin/env bash
# Experiment round 2 (branch native-perf-experiment only): runner facts, then compile all sources with the Vector API module.
set -e
mkdir -p target/bench
echo "RUNNER $(grep -m1 'model name' /proc/cpuinfo)"
echo "RUNNER nproc $(nproc)"
for f in avx avx2 fma avx512f avx512bw avx512vl avx512_vnni; do if grep -q -w "$f" /proc/cpuinfo; then echo "RUNNER flag $f"; fi; done
free -m | head -2
echo "TOOLCHAIN_HOME=${JAVA_HOME:-unset}"
"$JAVA_HOME/bin/java" -version 2>&1 | head -3
find src/main/java src/smoke/java -name '*.java' > target/bench/sources.txt
rm -rf "$CLASSES" && mkdir -p "$CLASSES"
"$JAVA_HOME/bin/javac" -d "$CLASSES" --add-modules jdk.incubator.vector @target/bench/sources.txt
echo "COMPILED $(find "$CLASSES" -name '*.class' | wc -l) classes"
