#!/usr/bin/env bash
# REMOVE WHEN: the arms of study 106 are read (read_results.py), the verdict is written on humemai/tiervec#106 and the result files are committed to humemai/tiervec (results/), or the study is dropped.
# Study 106: is the recall drop of the tier at 1,000,000 vectors (issue #106) the data or the build parameters? PLAN.md has the question, the pins, the arms and the rules, written before any run.
#
# usage: run.sh MODE
#   prep         the data of the primary set, on the host that holds the parquet shards (the laptop): DBpedia OpenAI text-embedding-3-large, first 768 coordinates, 999,000 vectors and 1,000
#                queries (real1m), then a seeded random subset of 100,000 vectors with the same queries (real100k), then an independent exact check of both. Run it with ALLOWED_CORES=14-15.
#   prep-deep    the control set (optional): the first 1,000,000 rows of DEEP (96 dimensions) and the 1,000 next rows as queries (deep1m), 384 MB. Needs the DEEP base file (STUDY_DEEP_NPY)
#   compile      compile the pinned harness (needs the repository with the pinned commit and the two JDK trees)
#   a0           real100k, M 16, efConstruction 200 (the tvm-h1m build)       -> results/a0-real100k-def.txt
#   a1           real1m, the same build                                       -> results/a1-real1m-def.txt
#   a2           real1m, --efc 400 --m 32 (the tvm-diag1m build d2)           -> results/a2-real1m-efc400-m32.txt
#   a3           real1m, --load rebuild (one segment; CONDITIONAL, README section 7) -> results/a3-real1m-rebuild.txt
#   b1           deep1m, the tvm-h1m build (optional control)                 -> results/b1-deep1m-def.txt
#   main         a0, a1, a2 one after the other; stops at the first arm that fails its gate
#   read         the results table (read_results.py) on stdout
# A result file is never overwritten: a second build of an arm needs RUN_TAG=-r2 (README section 4 allows one, for a1 only).
# Environment (all optional): ALLOWED_CORES (default 12-15: the cores this run may use, read back from the process, any other core refuses the run), XMX (13g), NICE (10), HOST_LABEL,
#   STUDY_W (work dir), STUDY_DATA, STUDY_SHARDS, STUDY_DEEP_NPY, TIERVEC_REPO, JDK25, JDK21, UV, RUN_TAG.
# Every JVM runs under nice. Pin the process from outside (the launcher I used is not published) and the run reads the allowed list back; it never pins itself.
set -u -o pipefail
H=$(cd "$(dirname "$0")" && pwd)
SHA=70b64f2e342430cb0c6389a187c35328219aca2e      # humemai/tiervec main of 2026-11-17: src/main equal to 70d5dee6c3 (tvm-h1m), plus the --data loader of #108
PREP_SHA256=4c6b1cf2041d5d55f6ae501666dec53b3fd61e016f76202398c2ff9054c821e3
W=${STUDY_W:-$HOME/study-106}
DATA=${STUDY_DATA:-$W/data}
SHARDS=${STUDY_SHARDS:-$W/shards}
DEEP_NPY=${STUDY_DEEP_NPY:-$W/deep_base.npy}
REPO=${TIERVEC_REPO:-$HOME/tiervec}
JDK25=${JDK25:-$W/jdk25}
JDK21=${JDK21:-$W/jdk21}
ALLOWED=${ALLOWED_CORES:-12-15}
XMX=${XMX:-13g}
NICE=${NICE:-10}
UV=${UV:-uv}
TAG=${RUN_TAG:-}
LABEL=${HOST_LABEL:-$(hostname -s)-cores-$ALLOWED}
MODE=${1:-}
mkdir -p "$W" "$H/results" "$H/logs"

die() { echo "run.sh: $*" >&2; exit 3; }

# expand_list "0,2,4-5" prints one core per line
expand_list() { awk -v s="$1" 'BEGIN { n = split(s, a, ","); for (i = 1; i <= n; i++) { if (a[i] ~ /-/) { split(a[i], r, "-"); for (c = r[1]; c <= r[2]; c++) print c } else if (a[i] != "") print a[i] } }'; }

# the run refuses to start on a core outside ALLOWED_CORES: the allowed list is read back from this process, never assumed
check_affinity() {
  local have c ok
  have=$(awk '/^Cpus_allowed_list/ { print $2 }' /proc/self/status)
  [ -n "$have" ] || die "cannot read Cpus_allowed_list"
  for c in $(expand_list "$have"); do
    ok=0
    for a in $(expand_list "$ALLOWED"); do [ "$a" = "$c" ] && ok=1; done
    [ "$ok" = 1 ] || { echo "run.sh: Cpus_allowed_list is $have, core $c is outside ALLOWED_CORES=$ALLOWED: start the run pinned to those cores" >&2; exit 5; }
  done
  echo "run.sh: Cpus_allowed_list=$have (inside $ALLOWED) label=$LABEL"
}

check_kit() {
  (cd "$H" && grep -v ' README.md$' SHA256SUMS | sha256sum -c --quiet -) || die "SHA256SUMS does not match: the kit was edited after it was sealed"
}

temp_c() { local t=unknown h; for h in /sys/class/hwmon/hwmon*; do [ "$(cat "$h/name" 2> /dev/null)" = coretemp ] && t=$(($(cat "$h/temp1_input") / 1000)); done; echo "$t"; }
throttle_total() { local s=0 c f any=0; for c in $(expand_list "$ALLOWED"); do f=/sys/devices/system/cpu/cpu$c/thermal_throttle/core_throttle_count; [ -r "$f" ] && { s=$((s + $(cat "$f"))); any=1; }; done; [ "$any" = 1 ] && echo "$s" || echo n/a; }

# tree SHA: a detached worktree of the pinned commit under the work dir (disk, never /tmp); prints its path
tree() {
  local wt=$W/wt-${SHA:0:10}
  git -C "$REPO" rev-parse --git-dir > /dev/null 2>&1 || die "no repository at $REPO (git clone https://github.com/humemai/tiervec.git $REPO), set TIERVEC_REPO"
  git -C "$REPO" cat-file -e "$SHA^{commit}" 2> /dev/null || die "$REPO does not have commit $SHA (git -C $REPO fetch origin)"
  [ -d "$wt" ] || git -C "$REPO" worktree add --detach "$wt" "$SHA" >&2 || die "git worktree add failed"
  [ "$(git -C "$wt" rev-parse HEAD)" = "$SHA" ] || die "worktree $wt is not at $SHA"
  [ -z "$(git -C "$wt" status --porcelain --untracked-files=no)" ] || die "worktree $wt has modified files"
  echo "$wt"
}

harness_classes() {
  [ -x "$JDK25/bin/java" ] && [ -x "$JDK21/bin/javac" ] || die "need JDK 25 at $JDK25 and the JDK 21 javac at $JDK21 (the tvm-h1m pair: 25.0.5 runs, 21.0.12 compiles)"
  HWT=$(tree) || exit 3
  [ -f "$HWT/src/bench/java/ai/humem/tiervec/RealData.java" ] || die "$SHA has no real data loader"
  HCLS=$W/classes-harness-${SHA:0:10}
  if [ ! -f "$HCLS/.sha" ] || [ "$(cat "$HCLS/.sha")" != "$SHA" ]; then
    rm -rf "$HCLS.new" && mkdir -p "$HCLS.new" || die "cannot make $HCLS.new"
    find "$HWT/src/main/java" "$HWT/src/bench/java" -name '*.java' > "$W/sources-harness.txt"
    nice -n "$NICE" "$JDK21/bin/javac" -nowarn -d "$HCLS.new" --add-modules jdk.incubator.vector @"$W/sources-harness.txt" || die "compile failed"
    echo "$SHA" > "$HCLS.new/.sha" && rm -rf "$HCLS" && mv "$HCLS.new" "$HCLS" || die "cannot place $HCLS"
  fi
}

# the pinned prep script, from git, checked by hash
pinned_prep() {
  git -C "$REPO" cat-file -e "$SHA^{commit}" 2> /dev/null || die "$REPO does not have commit $SHA"
  git -C "$REPO" show "$SHA:scripts/prep_real_dataset.py" > "$W/prep_real_dataset.py.new" || die "git show failed"
  [ "$(sha256sum "$W/prep_real_dataset.py.new" | cut -d' ' -f1)" = "$PREP_SHA256" ] || die "scripts/prep_real_dataset.py at $SHA has another sha256 than the pin"
  mv "$W/prep_real_dataset.py.new" "$W/prep_real_dataset.py"
}

meta_value() { awk -F= -v k="$2" '$1 == k { print $2 }' "$1/meta.properties" 2> /dev/null; }
need_data() { # need_data DIR COUNT DIM
  [ -f "$1/meta.properties" ] || die "$1 has no meta.properties: run prep first (or copy the prepared directory, README section 9)"
  [ "$(meta_value "$1" count)" = "$2" ] && [ "$(meta_value "$1" dim)" = "$3" ] || die "$1 has count $(meta_value "$1" count) and dim $(meta_value "$1" dim), expected $2 and $3"
}

env_header() { # env_header OUT
  {
    echo "ENV date=$(date -u +%FT%TZ) label=$LABEL host=$(hostname -s) study=study-106-20261123 sha=$SHA"
    echo "ENV cpus_allowed=$(awk '/^Cpus_allowed_list/ { print $2 }' /proc/self/status) allowed_expected=$ALLOWED nice=$NICE xmx=$XMX"
    echo "ENV cgroup_memory_max=$(cat "/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.max" 2> /dev/null || echo unknown)"
    echo "ENV mem_total_kib=$(awk '/^MemTotal/ { print $2 }' /proc/meminfo) mem_available_kib=$(awk '/^MemAvailable/ { print $2 }' /proc/meminfo) load=$(cut -d' ' -f1-3 /proc/loadavg)"
    echo "ENV pkg_c=$(temp_c) throttle_count_allowed_cores=$(throttle_total) governor=$(cat /sys/devices/system/cpu/cpu"$(expand_list "$ALLOWED" | head -1)"/cpufreq/scaling_governor 2> /dev/null || echo unknown)"
    echo "ENV ps_top_before:"
    ps -eo pid,psr,pcpu,etime,args --sort=-pcpu | head -9 | cut -c1-200 | sed 's/^/ENV   /'
  } > "$1"
}

sampler() { # sampler FILE: one line per 30 s until it is killed
  while true; do
    echo "$(date -u +%FT%TZ) pkg_c=$(temp_c) load=$(cut -d' ' -f1 /proc/loadavg) mem_available_kib=$(awk '/^MemAvailable/ { print $2 }' /proc/meminfo) throttle=$(throttle_total)" >> "$1"
    sleep 30
  done
}

COMMON=(--queries 1000 --reps 3 --ef-sweep 20,50,100,200,400,800,1600,3200 --shares 0.01 --scan-below 0 --build-threads 3 --conc-seconds 0)

# run_arm NAME DATADIR BENCH_ARGS...
run_arm() {
  local name=$1 data=$2 out spid rc; shift 2
  out=$H/results/$name$TAG.txt
  [ ! -e "$out" ] || die "$out exists: a result file is never overwritten (RUN_TAG=-r2 for a second build, README section 4)"
  env_header "$out.env"
  sampler "$out.samples" &
  spid=$!
  # the result file is the harness's own (its header names the hardware, the JDK, the commit, the pinned cores and every option); standard output goes to a log beside it
  ( cd "$HWT" && nice -n "$NICE" "$JDK25/bin/java" -Xmx"$XMX" --add-modules jdk.incubator.vector -cp "$HCLS" ai.humem.tiervec.Bench \
      --repo "$HWT" --host-label "$LABEL" --out "$out" --data "$data" "${COMMON[@]}" "$@" ) > "$H/logs/$name$TAG.stdout" 2>&1
  rc=$?
  kill "$spid" 2> /dev/null; wait "$spid" 2> /dev/null
  {
    echo "ENV ended=$(date -u +%FT%TZ) rc=$rc load=$(cut -d' ' -f1-3 /proc/loadavg) pkg_c=$(temp_c) throttle_count_allowed_cores=$(throttle_total)"
    echo "ENV ps_top_after:"
    ps -eo pid,psr,pcpu,etime,args --sort=-pcpu | head -9 | cut -c1-200 | sed 's/^/ENV   /'
  } >> "$out.env"
  echo "run.sh: $name rc=$rc out=$out"
  [ "$rc" = 0 ] && grep -q '^BENCH-END' "$out" && grep -q '^BENCH-SUMMARY status=OK failures=0' "$out"
}

run_one() { # run_one ARM
  check_kit
  check_affinity
  case $1 in
    a0) need_data "$DATA/real100k" 100000 768 ;;
    a1 | a2 | a3) need_data "$DATA/real1m" 999000 768 ;;
    b1) need_data "$DATA/deep1m" 1000000 96 ;;
    *) die "no such arm $1" ;;
  esac
  harness_classes
  case $1 in
    a0) run_arm a0-real100k-def "$DATA/real100k" ;;
    a1) run_arm a1-real1m-def "$DATA/real1m" ;;
    a2) run_arm a2-real1m-efc400-m32 "$DATA/real1m" --efc 400 --m 32 ;;
    a3) run_arm a3-real1m-rebuild "$DATA/real1m" --load rebuild ;;
    b1) run_arm b1-deep1m-def "$DATA/deep1m" ;;
  esac
}

case $MODE in
  prep)
    check_kit; check_affinity; pinned_prep
    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    [ -d "$SHARDS" ] || die "no shards at $SHARDS"
    nice -n 19 "$UV" run --python 3.12 "$H/prep_set.py" dbpedia "$SHARDS" "$DATA/real1m" --prep-script "$W/prep_real_dataset.py" --dim 768 --queries 1000 || exit 1
    nice -n 19 "$UV" run --python 3.12 "$H/prep_set.py" subset "$DATA/real1m" "$DATA/real100k" --n 100000 --seed 42 --prep-script "$W/prep_real_dataset.py" || exit 1
    nice -n 19 "$UV" run --python 3.12 "$H/prep_set.py" check "$DATA/real1m" || exit 1
    nice -n 19 "$UV" run --python 3.12 "$H/prep_set.py" check "$DATA/real100k" || exit 1
    echo "run.sh: prep done: $DATA/real1m $DATA/real100k"; du -sh "$DATA/real1m" "$DATA/real100k" ;;
  prep-deep)
    check_kit; check_affinity; pinned_prep
    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    [ -f "$DEEP_NPY" ] || die "no DEEP base file at $DEEP_NPY"
    nice -n 19 "$UV" run --python 3.12 "$H/prep_set.py" deep "$DEEP_NPY" "$DATA/deep1m" --prep-script "$W/prep_real_dataset.py" || exit 1
    nice -n 19 "$UV" run --python 3.12 "$H/prep_set.py" check "$DATA/deep1m" || exit 1
    du -sh "$DATA/deep1m" ;;
  compile) check_kit; check_affinity; harness_classes; echo "run.sh: harness classes in $HCLS (commit $SHA)" ;;
  a0 | a1 | a2 | a3 | b1) run_one "$MODE" ;;
  main)
    for arm in a0 a1 a2; do run_one "$arm" || { echo "run.sh: arm $arm failed its gate: stop, read the file before anything else" >&2; exit 1; }; done
    echo "run.sh: main chain done (a0, a1, a2)" ;;
  read) cd "$H" && "$UV" run --no-project python -I read_results.py "${@:2}" ;;
  *) echo "usage: run.sh prep|prep-deep|compile|a0|a1|a2|a3|b1|main|read" >&2; exit 2 ;;
esac
