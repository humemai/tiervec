#!/usr/bin/env bash
# Imports the release signing key into the GnuPG home named by GNUPGHOME and checks that it can sign. Used by the publish path of
# release-central.yml with the repository secrets, and by the dry run with a key it just made, so the dry run exercises this code.
#
# Input, from the environment only (never from an argument or a file):
#   GPG_PRIVATE_KEY  the armored private key
#   GPG_PASSPHRASE   its passphrase
# The key reaches gpg on standard input and nothing here prints it. The runner masks the secrets in the log.
set -euo pipefail

: "${GNUPGHOME:?GNUPGHOME must name a private GnuPG home}"
if [ -z "${GPG_PRIVATE_KEY:-}" ]; then
  echo "::error::GPG_PRIVATE_KEY is empty"
  exit 1
fi
if [ -z "${GPG_PASSPHRASE:-}" ]; then
  echo "::error::GPG_PASSPHRASE is empty"
  exit 1
fi

printf '%s\n' "$GPG_PRIVATE_KEY" | gpg --batch --import

count="$(gpg --batch --list-secret-keys --with-colons | grep -c '^sec:')"
if [ "$count" != "1" ]; then
  echo "::error::expected exactly one secret key after the import, found $count"
  exit 1
fi
fingerprint="$(gpg --batch --list-secret-keys --with-colons | awk -F: '/^fpr:/ { print $10; exit }')"
echo "signing key fingerprint: $fingerprint"

# Sign and verify a probe file now, so a wrong passphrase fails here and not after the build.
probe="$(mktemp "${RUNNER_TEMP:-/tmp}/probe.XXXXXX")"
echo probe > "$probe"
printf '%s' "$GPG_PASSPHRASE" | gpg --batch --pinentry-mode loopback --passphrase-fd 0 --detach-sign --output "$probe.asc" "$probe"
gpg --batch --verify "$probe.asc" "$probe"
rm -f "$probe" "$probe.asc"
echo "the key signs with the passphrase"
