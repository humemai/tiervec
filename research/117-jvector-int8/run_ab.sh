#!/usr/bin/env bash
# run_ab.sh of jv-int8-rc1-20261124/: the paired A/B of jvector native INT8 against FP32 on the RELEASE jvector 4.1.0-rc.1 (Maven Central jar, which contains PR #709, merge commit ee81b2e7d6).
# It repeats jv-int8-r934c13/run_ab_java.sh (itself the copy of the first run's, head 575dcee1b, with the third arm Z added): the same layer classes (src/, labels only), JMH settings, arm order, rounds,
# stopping rule, to_jsonl_java.py, aa_check.py and ab.py. What differs, and nothing else: (1) the jvector jar (jars/jvector-4.1.0-rc.1.jar); (2) the host: the benchmark host's compare lane, pinned by the Humem Worker
# (CPUAffinity), so there is no taskset and no night lock here, the Worker's lane gate does that; (3) the thermal wait and cooling of night_arm_gate are done by arm_gate below (70 C gate, arms of a round
# cooled to the first arm's start temperature within 2 C, as the laptop's night_lib did); (4) prep (untimed: graph build + calibration of the matched-recall setting on THIS build) runs first inside the
# same job, on the same cores; (5) no noise log (the laptop's sampler does not exist here): COLLISION cannot fire, foreign load is the Worker's lane gate (it halts the job) and its lane-evidence.json.
# Run by the Worker as: OUT=<label> bash ./run_ab.sh   (labels i8rc1n-<cells> and i8rc1p-<cells>, <cells> = sift128, d1536, d768, d256 joined by '_' in run order, or smoke*, optional -r2 suffix; n = the release jar's default provider, which is the
# NATIVE one (Highway .so inside the jar), p = the same jar forced to the Panama provider with -Djvector.vectorization_provider=panama, the provider of the posted 575dcee1b run)
# Writes $OUT/<cell>/{prep,work,ab}, $OUT/env.txt and, LAST, $OUT/analysis.txt (a waiter polls for it). Resumable: a restart of the same job skips finished steps.
# Exit: 0 valid (or QUICK smoke), 2 INVALID (A/A or collision), 3 WEAK (< 10 clean pairs at the cap), 4 an arm run failed, 5 not confined to a lane, 6 a pin or sum differs, 1/other: it broke.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export PATH="$HOME/.local/bin:$PATH"
OUT="${OUT:-}"
case "$OUT" in "" | . | .. | */*) echo "OUT must be a plain label, got '$OUT'" >&2; exit 2;; esac
. "$HERE/env_host.sh"
PYRUN=${PYRUN:-"uv run --no-project --python 3.12 python -I"}
py() { $PYRUN "$@"; }
QUICK=0; CELLS=; PROVIDER=; KEY=
case "$OUT" in
  i8rc1n-*) PROVIDER=native; KEY=${OUT#i8rc1n-};;
  i8rc1p-*) PROVIDER=panama; KEY=${OUT#i8rc1p-};;
  *) echo "unknown OUT label '$OUT'" >&2; exit 2;;
esac
# the label names the cells, in run order, joined by '_' (for example i8rc1n-sift128_d1536_d768_d256), with an optional '-r2' for the declared confirmatory rerun
KEY=${KEY%-r2}
case "$KEY" in
  smoke*) QUICK=1; CELLS="sift128-10000 dbpedia-openai3-1536-10000";;
  *) for k in ${KEY//_/ }; do case "$k" in
       sift128) CELLS="$CELLS sift128-100000";;
       d1536) CELLS="$CELLS dbpedia-openai3-1536-100000";;
       d768) CELLS="$CELLS dbpedia-openai3-768-100000";;
       d256) CELLS="$CELLS dbpedia-openai3-256-100000";;
       *) echo "unknown cell key '$k' in OUT label '$OUT'" >&2; exit 2;;
     esac; done;;
esac
[ -n "${CELLS// /}" ] || { echo "no cell in OUT label '$OUT'" >&2; exit 2; }
[ -z "${PARSE_ONLY:-}" ] || { echo "OUT=$OUT provider=$PROVIDER quick=$QUICK cells=$CELLS"; exit 0; }
# the provider of every arm: native = the jar's default (no property), panama = forced; run_cell checks that every arm reports it
EXPECT=NativeVectorizationProvider
if [ "$PROVIDER" = panama ]; then JFLAGS+=(-Djvector.vectorization_provider=panama); EXPECT=PanamaVectorizationProvider; fi
O="$HERE/$OUT"; mkdir -p "$O"
CPULIST=$(awk '/^Cpus_allowed_list/{print $2}' /proc/self/status)
cpus_expand() { local x a b; for x in ${1//,/ }; do case $x in *-*) a=${x%-*}; b=${x#*-}; seq "$a" "$b";; *) echo "$x";; esac; done; }
CPUS_ARR=($(cpus_expand "$CPULIST")); NT=${#CPUS_ARR[@]}
if [ "$QUICK" != 1 ] && [ "$(nproc)" -gt 4 ]; then
  echo "refused: the process may run on $(nproc) cores (Cpus_allowed_list $CPULIST); a timed run needs the lane's cores only" | tee "$O/REFUSED.txt" >&2; exit 5
fi
# ---- pins and sums: a study script, the jar, the classes' sources or a data file that differs from what was written down is a new study
( cd "$HERE" && sha256sum --quiet -c SHA256SUMS ) || { echo "REFUSED: a study file differs from SHA256SUMS" >&2; exit 6; }
[ "$(sha256sum "$INT8_JAR" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$HERE/jars/CENTRAL.sha256")" ] || { echo "REFUSED: the jvector jar is not the sha256 that Maven Central publishes" >&2; exit 6; }
( cd "$WHEEL_JARS" && sha256sum --quiet -c "$HERE/wheel-jars.sha256" ) || { echo "REFUSED: the wheel jars differ from wheel-jars.sha256" >&2; exit 6; }
[ -x "$JAVA" ] && [ -e "$CLS/vq/Layer1Main.class" ] && [ -e "$CLS/vq/JvInt8Arm.class" ] && ls "$JL"/target/lib/jmh-core-*.jar > /dev/null || { echo "REFUSED: JDK, classes or JMH libs missing" >&2; exit 6; }
for cell in $CELLS; do grep "  $cell/" "$HERE/data.sha256" | ( cd "$DATA" && sha256sum --quiet -c - ) || { echo "REFUSED: data of $cell differs from data.sha256" >&2; exit 6; }; done
HW=; for d in /sys/class/hwmon/hwmon*; do [ "$(cat "$d/name" 2> /dev/null)" = coretemp ] && HW=$d/temp1_input; done
temp_c() { if [ -r "$HW" ]; then echo $(( $(cat "$HW") / 1000 )); else echo 0; fi; }
thr() { local s=0 c f; for c in "${CPUS_ARR[@]}"; do f=/sys/devices/system/cpu/cpu$c/thermal_throttle/core_throttle_total_time_ms; [ -r "$f" ] && s=$((s + $(cat "$f"))); done; echo $s; }
{ printf 'started %s\nhost %s\nkernel %s\nout %s\ncells %s\nquick %s\nprovider %s (expected %s)\njflags %s\n' "$(date -Is)" "$(hostname)" "$(uname -srm)" "$OUT" "$CELLS" "$QUICK" "$PROVIDER" "$EXPECT" "${JFLAGS[*]}"
  printf 'nproc %s\nCpus_allowed_list %s\nHUMEM_HOST_LANE %s\nHUMEM_HOST_CPUS %s\nHUMEM_HOST_LANE_CLASS %s\nHUMEM_HOST_CELL_CLASS %s\n' "$(nproc)" "$CPULIST" "${HUMEM_HOST_LANE:-}" "${HUMEM_HOST_CPUS:-}" "${HUMEM_HOST_LANE_CLASS:-}" "${HUMEM_HOST_CELL_CLASS:-}"
  grep -m1 'model name' /proc/cpuinfo; printf 'governor %s\nno_turbo %s\nuv %s\n' "$(cat /sys/devices/system/cpu/cpu${CPUS_ARR[0]}/cpufreq/scaling_governor 2> /dev/null)" "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2> /dev/null)" "$(uv --version)"
  printf 'jdk %s\n' "$("$JAVA" -version 2>&1 | tr '\n' ' ')"; grep -E '^(JAVA_RUNTIME_VERSION|IMPLEMENTOR)=' "$JDK/release" 2> /dev/null
  printf 'jvector jar sha256 %s (%s)\n' "$(sha256sum "$INT8_JAR" | cut -d' ' -f1)" "$INT8_JAR"; printf 'wheel jars:\n'; cat "$HERE/wheel-jars.sha256"; printf 'kit sums:\n'; cat "$HERE/SHA256SUMS"
} > "$O/env.txt"

HEAP=${HEAP:-4g}; NQ=${NQ:-1000}; N=${NROUNDS:-12}; MAX=${MAXR:-20}; MINC=${MINCLEAN:-10}; METRICS=${METRICS:-"lat qps1 qpsN"}; JMH_EXTRA=${JMH_EXTRA:-"-f 1 -wi 3 -w 2s -i 5 -r 2s"}
if [ "$QUICK" = 1 ]; then NQ=200; N=2; MAX=2; MINC=1; JMH_EXTRA="-f 1 -wi 1 -w 1s -i 2 -r 1s"; export NIGHT_DRY=1; fi
GATE_C=${GATE_C:-70}; ARM_REF=; ARM_START_C=0
arm_gate() { local t s=$SECONDS
  [ "$QUICK" = 1 ] && { ARM_START_C=$(temp_c); return 0; }   # the smoke does not wait for the thermal gate (plumbing only)
  while t=$(temp_c); [ "$t" -gt "$GATE_C" ] && [ $((SECONDS - s)) -lt 600 ]; do sleep 10; done
  if [ -n "$ARM_REF" ]; then s=$SECONDS; while t=$(temp_c); [ "$t" -gt $((ARM_REF + 2)) ] && [ $((SECONDS - s)) -lt 120 ]; do sleep 3; done; fi
  ARM_START_C=$(temp_c); [ -n "$ARM_REF" ] || ARM_REF=$ARM_START_C; }
cur=; trap '[ -n "$cur" ] && kill "$cur" 2> /dev/null; exit 143' TERM INT
ARMS="jv-FP32 jv-INT8 jv-FP32-aa"

run1() { local arm=$1 metric=$2 k=$3 f=$A/$1.$2.$3 bm tu t ef rr extra jm rc t0 p0 m0
  [ -s "$f.json" ] && [ -s "$f.meta" ] && return 0
  case $metric in lat) bm=sample; tu=us; t=1;; qps1) bm=thrpt; tu=s; t=1;; qpsN) bm=thrpt; tu=s; t=$NT;; *) echo "metric $metric" >&2; return 2;; esac
  read -r ef rr <<< "$(l1_calib "$P/${arm%-aa}.query.log")"; extra="-p ef=$ef -p rerank=$rr -p nq=$NQ -bm $bm -tu $tu -t $t"
  arm_gate; t0=$(date +%s.%N); p0=$ARM_START_C; m0=$(thr)
  nice -n 10 "$JAVA" -Xmx$HEAP "${JFLAGS[@]}" -cp "$(l1_cp)" org.openjdk.jmh.Main QueryBench $JMH_EXTRA $extra -rf json -rff "$f.json" -p arm="${arm%-aa}" -p dir="$D" -p work="$W" \
    -jvmArgs "-Xmx$HEAP ${JFLAGS[*]}" > "$f.log" 2>&1 & cur=$!; wait $cur; rc=$?; cur=
  [ $rc -eq 0 ] && [ -s "$f.json" ] || { echo "$(date +%T) $arm $metric r$k FAILED rc=$rc (log $f.log)" >&2; rm -f "${f:?}.json"; return 2; }
  echo "{\"t0\": $t0, \"t1\": $(date +%s.%N), \"pkg_c\": $p0, \"throttle_ms\": $(( $(thr) - m0 )), \"ncores\": $NT, \"cpus\": \"$CPULIST\"}" > "$f.meta"; echo "$(date +%T) $arm $metric r$k done"; }
# order K: rotate the arm list by 5 per round, reverse it on odd rounds: every pair meets both orders over the rounds
order() { local k=$1 n i; local -a a=($TIMED); n=${#a[@]}; local -a o=(); for ((i = 0; i < n; i++)); do o+=("${a[$(( (i + k * 5) % n ))]}"); done
  if [ $((k % 2)) -eq 1 ]; then for ((i = n - 1; i >= 0; i--)); do echo "${o[$i]}"; done; else printf '%s\n' "${o[@]}"; fi; }
round() { local k=$1 metric arm; for metric in $METRICS; do ARM_REF=; for arm in $(order "$k"); do run1 "$arm" "$metric" "$k" || exit 4; done; done; }
analyse() { py "$HERE/to_jsonl_java.py" "$A" --clean --dim "$DIM" > "$A/ab.txt"; sed -n 's/^MINCLEAN //p' "$A/ab.txt"; }

run_cell() { local cell=$1 arm; D=$DATA/$cell; W=$O/$cell/work; P=$O/$cell/prep; A=$O/$cell/ab; mkdir -p "$W" "$P" "$A"
  DIM=$(sed -E 's/.*"dim": *([0-9]+).*/\1/' "$D/meta.json")
  local PH=4g; case $cell in sift128-*) PH=2g;; esac
  for arm in jv-FP32 jv-INT8; do
    [ -e "$P/$arm.done" ] && { echo "skip prep $arm (done)"; continue; }
    echo "$(date +%T) prep $arm"
    l1_java -Xmx$PH vq.Layer1Main build --arm "$arm" --dir "$D" --work "$W" --nq "$NQ" > "$P/$arm.build.log" 2>&1
    l1_java -Xmx$PH vq.Layer1Main query --arm "$arm" --dir "$D" --work "$W" --nq "$NQ" > "$P/$arm.query.log" 2>&1
    if grep -q 'BUILD ok' "$P/$arm.build.log" && grep -q 'QUERY ok' "$P/$arm.query.log" && ! grep -q '^SKIP' "$P/$arm.query.log"; then
      touch "$P/$arm.done"; echo "$(date +%T) ok $arm calib: $(l1_calib "$P/$arm.query.log") reached=$(l1_reached "$P/$arm.query.log")"
    else echo "FAILED prep $arm (logs in $P)" >&2; exit 4; fi
  done
  grep -h '^RESULT {' "$P"/*.log | sed 's/^RESULT //' > "$P/results.jsonl"
  TIMED=
  for arm in $ARMS; do
    [ "$(l1_reached "$P/${arm%-aa}.query.log")" = true ] || { echo "arm $arm never reached recall@10 0.95 (unmatched): not timed" >&2; continue; }
    TIMED="$TIMED $arm"
  done
  [ "$(echo $TIMED | wc -w)" -eq 3 ] || { echo "REFUSED: not all three arms are matched: '$TIMED'" >&2; exit 4; }
  { echo "cell $cell, cpus $CPULIST (nt $NT), heap $HEAP, JMH $JMH_EXTRA, metrics $METRICS, nq $NQ; $(uname -r); $(grep -m1 'model name' /proc/cpuinfo)"
    for arm in $TIMED; do echo "calib $arm ef,rerank = $(l1_calib "$P/${arm%-aa}.query.log")"; done
    for arm in $TIMED; do echo "arm $arm jvector jar sha256 $(sha256sum "$INT8_JAR" | cut -d' ' -f1) ($INT8_JAR)"; done; } > "$A/arms.txt"
  local k mc; for k in $(seq 0 "$N"); do round "$k"; done
  k=$((N + 1)); mc=$(analyse)
  while [ "${mc:-0}" -lt "$MINC" ] && [ "$k" -le "$MAX" ]; do round "$k"; k=$((k + 1)); mc=$(analyse); done
  analyse > /dev/null
  py "$HERE/aa_check.py" "$A"; VRC=$?
  # every arm of every run must have reported the expected vectorization provider (a native library that fails to load falls back to Panama silently); a mismatch makes the cell INVALID
  local bad; bad=$(grep -h '^INFO QueryBench' "$A"/*.log | grep -vc "\"vectorization\":\"$EXPECT\""); local seen; seen=$(grep -h '^INFO QueryBench' "$A"/*.log | grep -c "\"vectorization\":\"$EXPECT\"")
  echo "provider check: $seen QueryBench runs report $EXPECT, $bad report another" > "$A/provider.txt"
  if [ "$bad" -ne 0 ] || [ "$seen" -eq 0 ]; then VRC=2; echo "INVALID provider: $(cat "$A/provider.txt")" > "$A/validity.txt"; fi
  { echo "== cell $cell ($OUT)"; cat "$A/provider.txt"; cat "$A/arms.txt"; grep -E "^==|clean pairs|VERDICT|median" "$A/ab.txt" | head -200; echo "MINCLEAN $mc"; echo "VALIDITY exit $VRC: $(cat "$A/validity.txt" 2> /dev/null)"; } > "$O/$cell/analysis.txt"
  return $VRC
}
FINAL=0
for cell in $CELLS; do run_cell "$cell"; rc=$?; [ $rc -le $FINAL ] || FINAL=$rc; done
{ echo "finished $(date -Is) out $OUT final exit $FINAL"; for cell in $CELLS; do cat "$O/$cell/analysis.txt"; done; } > "$O/analysis.txt"
echo "ALLDONE $OUT exit $FINAL"; exit $FINAL
