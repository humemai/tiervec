#!/usr/bin/env bash
# Tests of verify-central-release.sh and check-central-absent.sh without Maven Central. A local HTTP server stands in for repo1 and for the
# key server, and a stub stands in for Maven. Every case makes a fabricated release (random files, checksums, signatures made with a
# throwaway key) and spoils one thing; the script must fail with the message of that thing, and the good set must pass. The cases check the
# message, not only the exit status, so a failure for the wrong reason does not pass.
#
# Needs: bash, curl, gpg, python3 (only as a static file server), md5sum, sha1sum, sha256sum, sha512sum.
# Set PYTHON_BIN to choose the Python (default python3). Nothing here reaches the network except 127.0.0.1.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
verify="$here/verify-central-release.sh"
absent="$here/check-central-absent.sh"
python_bin="${PYTHON_BIN:-python3}"
version="0.9.9"

tmp="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/central-tests.XXXXXX")"
server_pid=""
cleanup() {
  if [ -n "$server_pid" ]; then
    kill "$server_pid" 2> /dev/null || true
  fi
  GNUPGHOME="$tmp/signer" gpgconf --kill gpg-agent > /dev/null 2>&1 || true
  GNUPGHOME="$tmp/other" gpgconf --kill gpg-agent > /dev/null 2>&1 || true
  rm -rf "$tmp"
}
trap cleanup EXIT

passed=0
failed=0
ok() { passed=$((passed + 1)); echo "ok: $1"; }
bad() { failed=$((failed + 1)); echo "NOT OK: $1"; }
check() { # description, then a command that must succeed
  local description="$1"
  shift
  if "$@"; then ok "$description"; else bad "$description"; fi
}
log_has() { grep -q -- "$1" "$tmp/out"; }

# ---- the throwaway keys: "signer" is the release key, "other" signs what it must not ----
make_key() { # home
  mkdir -m 700 "$1"
  GNUPGHOME="$1" gpg --batch --pinentry-mode loopback --passphrase '' --quick-generate-key "tiervec test key <test@example.invalid>" ed25519 sign 1d > /dev/null 2>&1
}
fpr_of() { GNUPGHOME="$1" gpg --batch --list-keys --with-colons | awk -F: '$1 == "fpr" { print $10; exit }'; }
make_key "$tmp/signer"
make_key "$tmp/other"
signer_fpr="$(fpr_of "$tmp/signer")"
other_fpr="$(fpr_of "$tmp/other")"
if ! [[ "$signer_fpr" =~ ^[0-9A-F]{40}$ && "$other_fpr" =~ ^[0-9A-F]{40}$ && "$signer_fpr" != "$other_fpr" ]]; then echo "cannot make the test keys"; exit 1; fi

# ---- the local server: the site root has one directory per case ----
mkdir "$tmp/site"
cat > "$tmp/server.py" << 'PY'
import http.server, sys

root, portfile = sys.argv[1], sys.argv[2]

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=root, **kwargs)

    def log_message(self, *args):
        pass

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
with open(portfile, "w") as handle:
    handle.write(str(server.server_address[1]))
server.serve_forever()
PY
"$python_bin" -I "$tmp/server.py" "$tmp/site" "$tmp/port" &
server_pid=$!
for _ in $(seq 1 50); do
  [ -s "$tmp/port" ] && break
  sleep 0.2
done
[ -s "$tmp/port" ] || { echo "the test server did not start"; exit 1; }
port="$(cat "$tmp/port")"

# ---- fabricated releases ----
sign() { # home, file
  GNUPGHOME="$1" gpg --batch --yes --pinentry-mode loopback --passphrase '' --armor --detach-sign --output "$2.asc" "$2"
}
sums() { # file: md5 as the bare digest, sha1 and sha256 as "digest  name", like the files Central serves
  md5sum "$1" | awk '{ print $1 }' > "$1.md5"
  sha1sum "$1" | awk -v n="$(basename "$1")" '{ print $1 "  " n }' > "$1.sha1"
  sha256sum "$1" | awk '{ print $1 }' > "$1.sha256"
  sha512sum "$1" | awk '{ print $1 }' > "$1.sha512"
}
make_release() { # case name; creates $tmp/site/<case>/maven2/ai/humem/tiervec/<version> and $tmp/site/<case>/ks
  local dir="$tmp/site/$1/maven2/ai/humem/tiervec/$version" name
  mkdir -p "$dir" "$tmp/site/$1/ks/pks" "$tmp/built-$1"
  for name in "tiervec-$version.jar" "tiervec-$version-sources.jar" "tiervec-$version-javadoc.jar" "tiervec-$version.pom"; do
    head -c 4096 /dev/urandom > "$dir/$name"
    sign "$tmp/signer" "$dir/$name"
    sums "$dir/$name"
    case "$name" in *.jar) cp "$dir/$name" "$tmp/built-$1/$name" ;; esac
  done
  GNUPGHOME="$tmp/signer" gpg --batch --armor --export "$signer_fpr" > "$tmp/site/$1/ks/pks/lookup"
}
base() { printf 'http://127.0.0.1:%s/%s/maven2' "$port" "$1"; }
keys() { printf 'http://127.0.0.1:%s/%s/ks' "$port" "$1"; }

# ---- the Maven stub: it checks how it is called and then plays a resolution ----
cat > "$tmp/mvn-stub.sh" << 'STUB'
#!/usr/bin/env bash
# Environment: STUB_LOG (where the arguments go), STUB_SOURCE (the directory the "remote" files come from), STUB_MODE (fail, other-jar)
printf '%s\n' "$@" > "$STUB_LOG"
repo=""
settings=""
global=""
artifact=""
transitive=""
remote=""
previous=""
for argument in "$@"; do
  case "$argument" in
    -Dmaven.repo.local=*) repo="${argument#*=}" ;;
    -Dartifact=*) artifact="${argument#*=}" ;;
    -Dtransitive=*) transitive="${argument#*=}" ;;
    -DremoteRepositories=*) remote="${argument#*=}" ;;
  esac
  [ "$previous" = -s ] && settings="$argument"
  [ "$previous" = -gs ] && global="$argument"
  previous="$argument"
done
[ -f "$settings" ] && [ "$settings" = "$global" ] || { echo "stub: no isolated settings"; exit 21; }
if grep -q '<mirror>' "$settings"; then echo "stub: the settings file has a mirror"; exit 22; fi
[ -d "$repo" ] && [ -z "$(ls -A "$repo")" ] || { echo "stub: the local repository is not empty"; exit 23; }
[ "$artifact" = "ai.humem:tiervec:0.9.9" ] || { echo "stub: wrong artifact $artifact"; exit 24; }
[ "$transitive" = false ] || { echo "stub: transitive is not false"; exit 25; }
case "$remote" in central-check::default::http*) ;; *) echo "stub: wrong remote $remote"; exit 26 ;; esac
[ "$(pwd)" != "$STUB_PROJECT_DIR" ] || { echo "stub: ran inside the project"; exit 27; }
[ -z "$(ls -A .)" ] || { echo "stub: not run in an empty directory"; exit 28; }
if [ "${STUB_MODE:-}" = fail ]; then echo "stub: Could not find artifact ai.humem:tiervec:0.9.9"; exit 1; fi
target="$repo/ai/humem/tiervec/0.9.9"
mkdir -p "$target"
cp "$STUB_SOURCE/tiervec-0.9.9.pom" "$target/"
if [ "${STUB_MODE:-}" = other-jar ]; then head -c 4096 /dev/urandom > "$target/tiervec-0.9.9.jar"; else cp "$STUB_SOURCE/tiervec-0.9.9.jar" "$target/"; fi
STUB
chmod +x "$tmp/mvn-stub.sh"

# ---- the cases ----
run_verify() { # case, then more arguments for the script; output in $tmp/out, status in $rc
  local name="$1"
  shift
  rc=0
  STUB_LOG="$tmp/stub-$name.log" STUB_SOURCE="$tmp/site/$name/maven2/ai/humem/tiervec/$version" STUB_PROJECT_DIR="$here" \
    bash "$verify" --version "$version" --fingerprint "$signer_fpr" --base-url "$(base "$name")" --keyserver-url "$(keys "$name")" \
    --interval-seconds 1 --fetch-attempts 2 --key-attempts 2 --key-retry-seconds 1 --mvn "$tmp/mvn-stub.sh" "$@" > "$tmp/out" 2>&1 || rc=$?
}
expect_pass() { # description
  if [ "$rc" = 0 ] && grep -q "^verified: ai.humem:tiervec:$version is on Central" "$tmp/out"; then ok "$1"; else bad "$1 (exit $rc)"; sed 's/^/    | /' "$tmp/out"; fi
}
expect_fail() { # description, text the output must contain
  if [ "$rc" = 1 ] && grep -qF -- "$2" "$tmp/out" && ! grep -q '^verified:' "$tmp/out"; then ok "$1"; else bad "$1 (exit $rc, wanted: $2)"; sed 's/^/    | /' "$tmp/out"; fi
}

# 1. The good set passes, with the built jars compared, and Maven is called the way it must be.
make_release good
run_verify good --built-dir "$tmp/built-good" --summary "$tmp/summary.md"
expect_pass "a good set passes"
if grep -qx -- '-Dtransitive=false' "$tmp/stub-good.log" && grep -q 'dependency' "$tmp/stub-good.log" && grep -q 'central-check::default::http' "$tmp/stub-good.log"; then
  ok "Maven resolves ai.humem:tiervec with -Dtransitive=false from the given repository only"
else
  bad "the Maven stub was not called as required"
fi
if grep -q "Every .asc is valid and made by key" "$tmp/summary.md" && grep -q "^| .tiervec-0.9.9.jar. | .[0-9a-f]\{64\}. |$" "$tmp/summary.md"; then
  ok "the Markdown summary has the result and the sha256 of each file"
else
  bad "the summary file is wrong"
  cat "$tmp/summary.md" || true
fi
make_release nocompare
run_verify nocompare --no-built-compare
expect_pass "a good set passes without the built-jar comparison when it is switched off"

# 2. One flipped byte in the jar, the checksum files untouched.
make_release flipped
jar="$tmp/site/flipped/maven2/ai/humem/tiervec/$version/tiervec-$version.jar"
printf '\x00' | dd of="$jar" bs=1 seek=100 count=1 conv=notrunc 2> /dev/null
cp "$tmp/built-flipped/tiervec-$version.jar" "$tmp/flip-original"
if cmp -s "$jar" "$tmp/flip-original"; then printf '\xff' | dd of="$jar" bs=1 seek=100 count=1 conv=notrunc 2> /dev/null; fi
run_verify flipped --built-dir "$tmp/built-flipped"
expect_fail "a flipped byte in the jar fails the checksum" "checksum mismatch: tiervec-$version.jar.md5"

# 3. A missing .asc.
make_release noasc
rm "$tmp/site/noasc/maven2/ai/humem/tiervec/$version/tiervec-$version-sources.jar.asc"
run_verify noasc --built-dir "$tmp/built-noasc"
expect_fail "a missing .asc fails" "missing file: $(base noasc)/ai/humem/tiervec/$version/tiervec-$version-sources.jar.asc answered HTTP 404"

# 4. A wrong checksum file.
make_release wrongsum
printf '%s  tiervec-%s-javadoc.jar\n' "$(printf 'not the bytes' | sha1sum | awk '{ print $1 }')" "$version" > "$tmp/site/wrongsum/maven2/ai/humem/tiervec/$version/tiervec-$version-javadoc.jar.sha1"
run_verify wrongsum --built-dir "$tmp/built-wrongsum"
expect_fail "a wrong .sha1 fails" "checksum mismatch: tiervec-$version-javadoc.jar.sha1"

# 5. A bad signature: the pom changed, its checksums made again, the old signature kept.
make_release badsig
pom="$tmp/site/badsig/maven2/ai/humem/tiervec/$version/tiervec-$version.pom"
printf 'changed after signing' >> "$pom"
sums "$pom"
run_verify badsig --built-dir "$tmp/built-badsig"
expect_fail "a changed file under an old signature fails" "bad signature: gpg does not accept tiervec-$version.pom.asc"

# 6. A signature of another throwaway key: when the key server has only the release key, and when it has both.
make_release otherkey
pom="$tmp/site/otherkey/maven2/ai/humem/tiervec/$version/tiervec-$version.pom"
sign "$tmp/other" "$pom"
run_verify otherkey --built-dir "$tmp/built-otherkey"
expect_fail "a signature of an unknown key fails" "bad signature: gpg does not accept tiervec-$version.pom.asc"
GNUPGHOME="$tmp/other" gpg --batch --armor --export "$other_fpr" >> "$tmp/site/otherkey/ks/pks/lookup"
run_verify otherkey --built-dir "$tmp/built-otherkey"
expect_fail "a valid signature of another key fails" "signed by another key: tiervec-$version.pom.asc was made by $other_fpr"

# 7. The key server answers with another key than the one asked for.
make_release wrongkey
GNUPGHOME="$tmp/other" gpg --batch --armor --export "$other_fpr" > "$tmp/site/wrongkey/ks/pks/lookup"
run_verify wrongkey --built-dir "$tmp/built-wrongkey"
expect_fail "a key server answering with another key fails" "the key server answered with another key"

# 8. The key cannot be fetched.
make_release nokey
rm "$tmp/site/nokey/ks/pks/lookup"
run_verify nokey --built-dir "$tmp/built-nokey"
expect_fail "a key that cannot be fetched fails" "cannot fetch the signing key $signer_fpr"

# 9. The jar on Central is not the jar that was built (checksums and signatures are all fine).
make_release notbuilt
head -c 4096 /dev/urandom > "$tmp/built-notbuilt/tiervec-$version-sources.jar"
run_verify notbuilt --built-dir "$tmp/built-notbuilt"
expect_fail "Central differing from the built jar fails" "what is on Central is not what the workflow built: tiervec-$version-sources.jar"
cp "$tmp/site/notbuilt/maven2/ai/humem/tiervec/$version/tiervec-$version-sources.jar" "$tmp/built-notbuilt/"
rm "$tmp/built-notbuilt/tiervec-$version-javadoc.jar"
run_verify notbuilt --built-dir "$tmp/built-notbuilt"
expect_fail "a built jar that is missing fails" "no built file"

# 10. A 404 that never turns 200 fails at the deadline with the last answer.
mkdir -p "$tmp/site/never/maven2"
run_verify never --built-dir "$tmp/built-good" --deadline-seconds 3
expect_fail "a version that never appears fails at the deadline" "not on Central after"
check "the deadline message names the last answer" log_has "last answer: HTTP 404 from $(base never)/ai/humem/tiervec/$version/tiervec-$version.pom"

# 11. A 404 that turns 200 after two seconds passes: the polling works.
make_release late
mv "$tmp/site/late/maven2/ai/humem/tiervec/$version" "$tmp/late-hidden"
( sleep 2; mv "$tmp/late-hidden" "$tmp/site/late/maven2/ai/humem/tiervec/$version" ) &
late_pid=$!
run_verify late --built-dir "$tmp/built-late" --deadline-seconds 30
wait "$late_pid"
expect_pass "a version that appears while polling passes"
check "the poll said that it was waiting" log_has '^waiting for Central'

# 12. Maven fails, and Maven resolves another jar.
make_release mvnfail
STUB_MODE=fail run_verify mvnfail --built-dir "$tmp/built-mvnfail"
expect_fail "a failing Maven resolution fails" "Maven could not resolve ai.humem:tiervec:$version from Central in a fresh local repository"
make_release mvnother
STUB_MODE=other-jar run_verify mvnother --built-dir "$tmp/built-mvnother"
expect_fail "Maven resolving another jar fails" "resolved jar differs"

# 13. The script refuses to run without a decision about the built-jar comparison, and with a bad version or fingerprint.
rc=0
bash "$verify" --version "$version" --fingerprint "$signer_fpr" > "$tmp/out" 2>&1 || rc=$?
check "no --built-dir and no --no-built-compare is refused" [ "$rc" = 2 ] && check "the refusal says what to give" log_has 'give --built-dir'
rc=0
bash "$verify" --version 0.1.0-SNAPSHOT --fingerprint "$signer_fpr" --no-built-compare > "$tmp/out" 2>&1 || rc=$?
check "a SNAPSHOT version is refused (exit 2)" [ "$rc" = 2 ] && check "the refusal says why" log_has 'must be a release version'
rc=0
bash "$verify" --version "$version" --fingerprint 5AE981B2 --no-built-compare > "$tmp/out" 2>&1 || rc=$?
check "a short fingerprint is refused (exit 2)" [ "$rc" = 2 ] && check "the refusal says why" log_has '40 hex digits'

# 14. The absence check: 404 goes on, 200 refuses, no answer refuses.
check_absent() { # expected exit, expected text, description, base URL
  rc=0
  bash "$absent" --version "$version" --base-url "$4" --attempts 2 --retry-seconds 1 > "$tmp/out" 2>&1 || rc=$?
  if [ "$rc" = "$1" ] && grep -qF -- "$2" "$tmp/out"; then ok "$3"; else bad "$3 (exit $rc, wanted $1 and: $2)"; sed 's/^/    | /' "$tmp/out"; fi
}
check_absent 0 "is not on Central" "a version Central answers 404 for may be published" "$(base never)"
check_absent 1 "is already on Central" "a version Central answers 200 for is refused" "$(base good)"
check_absent 2 "cannot tell whether" "no answer from Central is refused, not taken as absent" "http://127.0.0.1:1/maven2"

echo
echo "$passed passed, $failed failed"
[ "$failed" = 0 ] && [ "$passed" -ge 29 ]
