#!/usr/bin/env bash
# Refuses to publish a version that is already on Maven Central, so a re-run of the publish run cannot push a duplicate.
# Exit 0: Central answers 404 for the version directory (it is not there). Exit 1: it answers 200 (it is there). Exit 2: no clear answer
# after the retries; the publish stops, because "cannot tell" must not mean "go on".
# Usage: check-central-absent.sh --version X.Y.Z [--base-url URL] [--attempts N] [--retry-seconds N]
set -euo pipefail

version=""
base_url="https://repo1.maven.org/maven2"
attempts=3
retry_seconds=5
while [ $# -gt 0 ]; do
  case "$1" in
    --version) version="${2:?--version needs a value}"; shift 2 ;;
    --base-url) base_url="${2:?--base-url needs a value}"; shift 2 ;;
    --attempts) attempts="${2:?--attempts needs a value}"; shift 2 ;;
    --retry-seconds) retry_seconds="${2:?--retry-seconds needs a value}"; shift 2 ;;
    *) echo "::error::unknown argument $1"; exit 2 ;;
  esac
done
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "::error::--version must be a release version such as 0.1.1, got '$version'"; exit 2; }

url="${base_url%/}/ai/humem/tiervec/$version/"
code=000
for attempt in $(seq 1 "$attempts"); do
  code="$(curl -sS --connect-timeout 20 --max-time 60 -o /dev/null -w '%{http_code}' "$url" 2> /dev/null)" || true
  code="${code:-000}"
  case "$code" in
    404)
      echo "ai.humem:tiervec:$version is not on Central ($url answers 404): the publish goes on"
      exit 0 ;;
    200)
      echo "::error::ai.humem:tiervec:$version is already on Central ($url answers 200). A version cannot be published twice or changed; the fix is the next patch version."
      exit 1 ;;
  esac
  echo "attempt $attempt of $attempts: $url answered HTTP $code"
  if [ "$attempt" -lt "$attempts" ]; then
    sleep "$retry_seconds"
  fi
done
echo "::error::cannot tell whether ai.humem:tiervec:$version is on Central (last answer HTTP $code from $url); nothing is published"
exit 2
