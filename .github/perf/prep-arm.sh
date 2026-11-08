#!/usr/bin/env bash
# Experiment round 3 (branch native-perf-experiment only): runner facts for Linux arm64 and macOS arm64, then compile all sources with the Vector API module.
set -e
mkdir -p target/bench
echo "RUNNER uname $(uname -sm)"
if [ "$(uname -s)" = Darwin ]; then
  echo "RUNNER model name $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  echo "RUNNER ncpu $(sysctl -n hw.ncpu)"
  sysctl -a 2>/dev/null | grep -i -E '^hw\.optional\.(neon|arm\.FEAT_(FP16|BF16|DotProd|I8MM|SVE|SME)|AdvSIMD)' | sed 's/^/RUNNER sysctl /' || true
  sw_vers | sed 's/^/RUNNER /'
else
  echo "RUNNER $(lscpu | grep -m1 -i 'model name' || echo 'model name unknown')"
  echo "RUNNER nproc $(nproc)"
  echo "RUNNER features $(grep -m1 -i '^Features' /proc/cpuinfo | cut -c1-400)"
  for f in asimd fphp asimdhp asimddp sve sve2 svebf16 i8mm bf16; do if grep -m1 '^Features' /proc/cpuinfo | grep -q -w "$f"; then echo "RUNNER flag $f"; fi; done
  echo "RUNNER cpu part $(grep -m1 -i 'CPU part' /proc/cpuinfo)"
  free -m | head -2
fi
echo "TOOLCHAIN_HOME=${JAVA_HOME:-unset}"
"$JAVA_HOME/bin/java" -version 2>&1 | head -3
"$JAVA_HOME/bin/java" -XX:+PrintFlagsFinal -version 2>/dev/null | grep -E ' (UseSVE|MaxVectorSize|UseNeon|UseSIMDForMemoryOps|EnableVectorSupport|UseVectorStubs) ' | sed 's/^/RUNNER jvmflag /' || true
find src/main/java src/smoke/java -name '*.java' > target/bench/sources.txt
rm -rf "$CLASSES" && mkdir -p "$CLASSES"
"$JAVA_HOME/bin/javac" -d "$CLASSES" --add-modules jdk.incubator.vector @target/bench/sources.txt
echo "COMPILED $(find "$CLASSES" -name '*.class' | wc -l) classes"
