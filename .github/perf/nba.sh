#!/usr/bin/env bash
# Experiment round 3 (branch native-perf-experiment only): build ONE native image and run it, portable to Linux and macOS. Never fails the step.
# usage: nba.sh TAG MAINCLASS NAME [native-image flags...]   (needs CLASSES, runs from the repo root)
tag=$1; main=$2; name=$3; shift 3
out=target/bench
bin="$out/$tag-$name"
log="$out/$tag-$name-build.log"
echo "::group::build $tag $name"
echo "BUILD tag=$tag name=$name main=$main flags=$*"
start=$(date +%s)
if native-image -cp "$CLASSES" --no-fallback --verbose "$@" -o "$bin" "com.humemai.vec.$main" > "$log" 2>&1; then
  echo "BUILD-OK tag=$tag name=$name secs=$(( $(date +%s) - start )) size=$(wc -c < "$bin" | tr -d ' ')"
  grep -i -E 'vector|intrinsic|species|march|warning|unsupported|architecture|SVE|NEON|aarch64' "$log" | cut -c1-300 | sort | uniq -c | sort -rn | head -40 | sed "s/^/BUILDLINE $tag $name /"
  echo "::endgroup::"
  "$bin" "$tag/$name" 2>&1
  echo "RUN-EXIT tag=$tag name=$name rc=$?"
else
  echo "BUILD-FAILED tag=$tag name=$name secs=$(( $(date +%s) - start ))"
  grep -i -E 'error|vector|march|unsupported|unrecognized' "$log" | cut -c1-300 | head -30 | sed "s/^/BUILDLINE $tag $name /"
  tail -15 "$log" | cut -c1-300 | sed "s/^/BUILDTAIL $tag $name /"
  echo "::endgroup::"
fi
exit 0
