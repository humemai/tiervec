#!/usr/bin/env bash
# Verifies a published ai.humem:tiervec version on Maven Central. Used by the publish run of release-central.yml after the Central Portal
# has published the deployment, and by verify-central.yml for a version that is already out. It changes nothing and publishes nothing.
#
# In order:
#   1. polls until tiervec-<version>.pom is served under the base URL (the sync from the Portal to repo1 takes minutes);
#   2. downloads the jar, the sources jar, the javadoc jar and the pom, each with its .asc, .md5, .sha1, .sha256 (and .sha512 when served);
#   3. checks every checksum file against the bytes;
#   4. fetches the public signing key from a key server by its fingerprint and checks every .asc with it, signed by that key and no other;
#   5. compares the three jars with the jars the workflow built (--built-dir), so what is on Central is what was built;
#   6. resolves ai.humem:tiervec:<version> in a fresh empty local Maven repository, with an isolated settings file (no mirrors).
# Any failure exits 1 with its own message. A failure never unpublishes anything: a Central version cannot be taken back.
#
# Usage: verify-central-release.sh --version X.Y.Z --fingerprint FPR (--built-dir DIR | --no-built-compare) [options]
# Options (the defaults are the real services; the tests point them at a local server):
#   --base-url URL           repository root, default https://repo1.maven.org/maven2
#   --keyserver-url URL      key server (HKP over HTTP), default https://keyserver.ubuntu.com
#   --deadline-seconds N     how long to wait for the pom, default 3600
#   --interval-seconds N     pause between polls and between download retries, default 30
#   --fetch-attempts N       tries per file that answers 404 after the pom was served, default 10
#   --key-attempts N         tries to fetch the key, default 3 (--key-retry-seconds N apart, default 10)
#   --summary FILE           append a Markdown result to FILE (the workflow passes $GITHUB_STEP_SUMMARY)
#   --mvn CMD                the Maven command, default mvn (the tests use a stub)
#   --dependency-plugin GAV  the plugin that resolves, default org.apache.maven.plugins:maven-dependency-plugin:3.11.0
set -euo pipefail

version=""
fingerprint=""
base_url="https://repo1.maven.org/maven2"
keyserver_url="https://keyserver.ubuntu.com"
built_dir=""
no_built_compare=false
deadline_seconds=3600
interval_seconds=30
fetch_attempts=10
key_attempts=3
key_retry_seconds=10
summary_file=""
mvn_cmd="mvn"
dependency_plugin="org.apache.maven.plugins:maven-dependency-plugin:3.11.0"

usage_error() {
  echo "::error::$*"
  sed -n '/^# Usage:/,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version) version="${2:?--version needs a value}"; shift 2 ;;
    --fingerprint) fingerprint="${2:?--fingerprint needs a value}"; shift 2 ;;
    --base-url) base_url="${2:?--base-url needs a value}"; shift 2 ;;
    --keyserver-url) keyserver_url="${2:?--keyserver-url needs a value}"; shift 2 ;;
    --built-dir) built_dir="${2:?--built-dir needs a value}"; shift 2 ;;
    --no-built-compare) no_built_compare=true; shift ;;
    --deadline-seconds) deadline_seconds="${2:?--deadline-seconds needs a value}"; shift 2 ;;
    --interval-seconds) interval_seconds="${2:?--interval-seconds needs a value}"; shift 2 ;;
    --fetch-attempts) fetch_attempts="${2:?--fetch-attempts needs a value}"; shift 2 ;;
    --key-attempts) key_attempts="${2:?--key-attempts needs a value}"; shift 2 ;;
    --key-retry-seconds) key_retry_seconds="${2:?--key-retry-seconds needs a value}"; shift 2 ;;
    --summary) summary_file="${2:?--summary needs a value}"; shift 2 ;;
    --mvn) mvn_cmd="${2:?--mvn needs a value}"; shift 2 ;;
    --dependency-plugin) dependency_plugin="${2:?--dependency-plugin needs a value}"; shift 2 ;;
    -h|--help) usage_error "help" ;;
    *) usage_error "unknown argument $1" ;;
  esac
done

[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || usage_error "--version must be a release version such as 0.1.1, got '$version'"
[[ "$fingerprint" =~ ^[0-9A-Fa-f]{40}$ ]] || usage_error "--fingerprint must be the 40 hex digits of the signing key, got '$fingerprint'"
fingerprint="${fingerprint^^}"
for number in "$deadline_seconds" "$interval_seconds" "$fetch_attempts" "$key_attempts" "$key_retry_seconds"; do
  [[ "$number" =~ ^[0-9]+$ ]] || usage_error "a number option has the value '$number'"
done
if [ -z "$built_dir" ] && [ "$no_built_compare" = false ]; then
  usage_error "give --built-dir (the jars the workflow built) or --no-built-compare (a version that is already out)"
fi
base_url="${base_url%/}"
keyserver_url="${keyserver_url%/}"

work="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/central-verify.XXXXXX")"
# This script's own GnuPG home: the public key is imported here and nowhere else, never into the home of the signing step.
GNUPGHOME="$work/gnupg"
mkdir -m 700 "$GNUPGHOME"
export GNUPGHOME
cleanup() {
  gpgconf --kill gpg-agent > /dev/null 2>&1 || true
  rm -rf "$work"
}
trap cleanup EXIT

fail() {
  echo "::error::verification of ai.humem:tiervec:$version failed: $*"
  if [ -n "$summary_file" ]; then
    { echo "### Central verification FAILED for $version"; echo; echo "$*"; echo; echo "Nothing was unpublished: a version on Central cannot be taken back. The fix is the next patch release (docs/RELEASING.md)."; } >> "$summary_file"
  fi
  exit 1
}

http_code() { # url, output file or /dev/null; prints the HTTP status, 000 when there is no answer
  local code
  code="$(curl -sS --connect-timeout 20 --max-time 300 -o "$2" -w '%{http_code}' "$1" 2> "$work/curl.err")" || true
  printf '%s' "${code:-000}"
}

dir_url="$base_url/ai/humem/tiervec/$version"
pom_url="$dir_url/tiervec-$version.pom"

# 1. Wait for the sync.
started=$SECONDS
while :; do
  code="$(http_code "$pom_url" /dev/null)"
  [ "$code" = 200 ] && break
  last="HTTP $code from $pom_url"
  [ "$code" = 000 ] && last="$last ($(tr '\n' ' ' < "$work/curl.err"))"
  waited=$((SECONDS - started))
  if [ "$waited" -ge "$deadline_seconds" ]; then
    fail "the version is not on Central after $waited seconds; last answer: $last"
  fi
  echo "waiting for Central after $waited of $deadline_seconds seconds; last answer: $last"
  sleep "$interval_seconds"
done
echo "Central serves $pom_url after $((SECONDS - started)) seconds"

# 2. Download.
dl="$work/central"
mkdir "$dl"
artifacts=("tiervec-$version.jar" "tiervec-$version-sources.jar" "tiervec-$version-javadoc.jar" "tiervec-$version.pom")
fetch() { # file name, "required" or "optional"; returns 1 only for an optional file that is not there
  local name="$1" attempt code
  for attempt in $(seq 1 "$fetch_attempts"); do
    code="$(http_code "$dir_url/$name" "$dl/$name")"
    [ "$code" = 200 ] && return 0
    if [ "$code" != 404 ] || [ "$attempt" -ge "$fetch_attempts" ]; then
      break
    fi
    sleep "$interval_seconds"
  done
  rm -f "$dl/$name"
  [ "$2" = optional ] && [ "$code" = 404 ] && return 1
  fail "missing file: $dir_url/$name answered HTTP $code"
}
for artifact in "${artifacts[@]}"; do
  for ext in "" .asc .md5 .sha1 .sha256; do
    fetch "$artifact$ext" required
  done
  fetch "$artifact.sha512" optional || true
done
echo "downloaded ${#artifacts[@]} artifacts with their signatures and checksums"

# 3. Checksums.
sum_of() { # file, tool; prints the lowercase hex digest of the bytes
  "$2" "$1" | awk '{ print tolower($1); exit }'
}
check_sum() { # artifact, extension, tool
  local file="$dl/$1" expected actual
  expected="$(awk '{ print tolower($1); exit }' "$file.$2")"
  actual="$(sum_of "$file" "$3")"
  if [ -z "$expected" ] || [ "$expected" != "$actual" ]; then
    fail "checksum mismatch: $1.$2 says '$expected', the bytes of $1 hash to $actual"
  fi
}
for artifact in "${artifacts[@]}"; do
  check_sum "$artifact" md5 md5sum
  check_sum "$artifact" sha1 sha1sum
  check_sum "$artifact" sha256 sha256sum
  if [ -f "$dl/$artifact.sha512" ]; then
    check_sum "$artifact" sha512 sha512sum
  fi
done
echo "checksums match the bytes (md5, sha1, sha256, and sha512 where served)"

# 4. Signatures with the key from the key server.
key_file="$work/signing-key.asc"
fetched=false
for attempt in $(seq 1 "$key_attempts"); do
  code="$(http_code "$keyserver_url/pks/lookup?op=get&options=mr&search=0x$fingerprint" "$key_file")"
  if [ "$code" = 200 ] && grep -q 'BEGIN PGP PUBLIC KEY BLOCK' "$key_file"; then
    fetched=true
    break
  fi
  echo "key server, attempt $attempt of $key_attempts: HTTP $code"
  if [ "$attempt" -lt "$key_attempts" ]; then
    sleep "$key_retry_seconds"
  fi
done
if [ "$fetched" = false ]; then
  fail "cannot fetch the signing key $fingerprint from $keyserver_url (last answer HTTP $code); Central needs the key on a key server too"
fi
gpg --batch --import "$key_file" 2> "$work/import.err" || fail "gpg cannot import the key from the key server: $(tr '\n' ' ' < "$work/import.err" | head -c 300)"
primary="$(gpg --batch --list-keys --with-colons | awk -F: '$1 == "pub" { want = 1; next } $1 == "fpr" && want { print $10; want = 0 }')"
if ! printf '%s\n' "$primary" | grep -qx "$fingerprint"; then
  fail "the key server answered with another key: wanted $fingerprint, got $(printf '%s' "$primary" | tr '\n' ' ')"
fi
for artifact in "${artifacts[@]}"; do
  if ! status="$(gpg --batch --status-fd 1 --verify "$dl/$artifact.asc" "$dl/$artifact" 2> "$work/gpg.err")"; then
    fail "bad signature: gpg does not accept $artifact.asc for $artifact ($(tr '\n' ' ' < "$work/gpg.err" | head -c 300))"
  fi
  # VALIDSIG <signing key fpr> ... <primary key fpr>: the last field is the primary key.
  signer="$(printf '%s\n' "$status" | awk '$1 == "[GNUPG:]" && $2 == "VALIDSIG" { print toupper($3) " " toupper($NF); exit }')"
  if [ -z "$signer" ]; then
    fail "bad signature: gpg gave no valid signature for $artifact.asc"
  fi
  if ! printf '%s\n' "$signer" | tr ' ' '\n' | grep -qx "$fingerprint"; then
    fail "signed by another key: $artifact.asc was made by $signer, not by $fingerprint"
  fi
done
echo "every signature is valid and made by $fingerprint"

# 5. What is on Central is what was built.
built_note="not compared (a version that was built by an earlier run)"
if [ "$no_built_compare" = false ]; then
  for artifact in "${artifacts[@]}"; do
    case "$artifact" in *.jar) ;; *) continue ;; esac
    if [ ! -f "$built_dir/$artifact" ]; then
      fail "no built file $built_dir/$artifact to compare with"
    fi
    built_sum="$(sum_of "$built_dir/$artifact" sha256sum)"
    central_sum="$(sum_of "$dl/$artifact" sha256sum)"
    if [ "$built_sum" != "$central_sum" ]; then
      fail "what is on Central is not what the workflow built: $artifact built sha256 $built_sum, Central sha256 $central_sum"
    fi
  done
  built_note="the three jars are byte for byte the jars the workflow built"
fi
echo "$built_note"

# 6. A consumer resolves it from Central alone: a fresh empty local repository, an empty directory (no project, so no repository of a
# pom), and an isolated settings file that is also the global one, so no mirror or profile of the host applies.
repo="$work/fresh-local-repository"
empty="$work/no-project"
mkdir "$repo" "$empty"
settings="$work/isolated-settings.xml"
cat > "$settings" << XML
<settings xmlns="http://maven.apache.org/SETTINGS/1.2.0">
  <localRepository>$repo</localRepository>
  <mirrors/>
  <profiles/>
  <activeProfiles/>
</settings>
XML
if ! (cd "$empty" && "$mvn_cmd" -B -ntp -s "$settings" -gs "$settings" -Dmaven.repo.local="$repo" \
  "$dependency_plugin:get" "-Dartifact=ai.humem:tiervec:$version" -Dtransitive=false "-DremoteRepositories=central-check::default::$base_url") > "$work/mvn.log" 2>&1; then
  tail -n 30 "$work/mvn.log"
  fail "Maven could not resolve ai.humem:tiervec:$version from Central in a fresh local repository"
fi
resolved="$repo/ai/humem/tiervec/$version/tiervec-$version.jar"
if [ ! -f "$resolved" ]; then
  tail -n 30 "$work/mvn.log"
  fail "Maven reported success but there is no $resolved in the fresh local repository"
fi
if [ "$(sum_of "$resolved" sha256sum)" != "$(sum_of "$dl/tiervec-$version.jar" sha256sum)" ]; then
  fail "resolved jar differs: the jar Maven resolved is not the jar Central serves"
fi
echo "Maven resolved ai.humem:tiervec:$version from Central into a fresh local repository"

echo "verified: ai.humem:tiervec:$version is on Central, intact, signed by $fingerprint and resolvable"
if [ -n "$summary_file" ]; then
  {
    echo "### Central verification for ai.humem:tiervec:$version"
    echo
    echo "The version is on Maven Central and passed every check."
    echo
    echo "- Served by $base_url after $((SECONDS - started)) seconds of waiting."
    echo "- Checksums (md5, sha1, sha256) match the bytes of the jar, sources jar, javadoc jar and pom."
    echo "- Every .asc is valid and made by key \`$fingerprint\`, fetched from $keyserver_url."
    echo "- Jars: $built_note."
    echo "- \`mvn $dependency_plugin:get\` resolved the artifact in a fresh local repository with an isolated settings file."
    echo
    echo "| File | sha256 |"
    echo "| --- | --- |"
    for artifact in "${artifacts[@]}"; do
      echo "| \`$artifact\` | \`$(sum_of "$dl/$artifact" sha256sum)\` |"
    done
  } >> "$summary_file"
fi
