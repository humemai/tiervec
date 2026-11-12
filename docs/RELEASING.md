# Releasing

This page is the rule for version numbers and the steps of a release. The first release is `0.1.0`. Between releases the version on `main` is the next `-SNAPSHOT`.

## Version numbers

- A release version is `0.MINOR.PATCH` while the API is not stable. At 1.0 it becomes `MAJOR.MINOR.PATCH`.
- A breaking change to the supported surface needs a new MINOR version. A release that has none can be a new PATCH version. The rule and the meaning of "breaking" are in [API.md](API.md#stability-policy-for-0x).
- Between releases `main` carries a `-SNAPSHOT` version, as Maven expects. After `0.1.0`, `main` is `0.1.1-SNAPSHOT`. When the first breaking change lands, `main` moves to `0.2.0-SNAPSHOT`.
- There is no `.devN`, `aN`, `bN` or `rcN` form. Maven does not order `0.1.0-dev1` before `0.1.0`: it sorts an unknown qualifier after the release. A pre-release is a `-SNAPSHOT`.
- A tag is the bare version, for example `0.1.0`, with no `v`.
- The version has one source: the `<version>` line in `pom.xml`.

## What a release has

- A tag on the commit of `main` that carries the release version.
- A section in [CHANGELOG.md](../CHANGELOG.md) with the version and the date.
- A GitHub release whose notes are that section.
- A documentation version `<version>` with the alias `latest` at <https://docs.humem.ai/tiervec/>. The tag push publishes it (workflow `deploy-docs.yml`). A push to `main` publishes only `dev`.
- The artifact `ai.humem:tiervec:<version>` on Maven Central: the jar, a sources jar and a javadoc jar, each signed. Publishing the GitHub release starts everything (workflow `release-central.yml`): the upload, the publication by the Central Portal, and a check of the result on `repo1.maven.org`. Nobody presses a button in the portal. The details are in the section "Maven Central" below ([#62](https://github.com/humemai/tiervec/issues/62)).

## The changelog

[CHANGELOG.md](../CHANGELOG.md) follows the "Keep a Changelog" layout: a section `Unreleased` at the top, then one section per release, newest first. The groups in a section are `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed` and `Security`.

A pull request that changes the supported surface, the file format, or a documented guarantee adds its line to `Unreleased` in the same pull request. A breaking change says so and says what to change. A change in the file format says which older readers refuse the new files ([SEGMENT-FORMAT.md](SEGMENT-FORMAT.md#versioning-and-compatibility)).

## Steps

Only the maintainer tags and publishes a release.

1. Check that the latest commit on `main` is green: every check of its run is `SUCCESS` (a skipped check is fine). Do not release from a red or running `main`.
2. Create the branch `release/<version>` from that commit.
3. In `pom.xml`, change `<version>` to the release version (no `-SNAPSHOT`).
4. In `CHANGELOG.md`, rename `Unreleased` to `<version> - YYYY-MM-DD`, add an empty `Unreleased` above it, and read the entry once against `git log <previous tag>..HEAD`.
5. Update `docs/STATUS.md` and the README status table where the release changes what they say (for example the line "A release: none").
6. Run the Central dry run on the branch and read its result (see "Maven Central" below):

   ```
   gh workflow run release-central.yml --ref release/<version> -f dry-run=true
   ```

7. Open a pull request named `Release <version>`, wait for green, and merge it.
8. Tag the merge commit and push the tag. The tag is the bare version:

   ```
   git tag <version> <merge-sha>
   git push origin <version>
   ```

   The tag push starts the workflow `Deploy docs`. It publishes the documentation as `<version>` with the alias `latest`, and `latest` is the default of the site.

9. Publish the GitHub release. The notes are the changelog section of this version:

   ```
   gh release create <version> --title <version> --notes-file <file with the section>
   ```

   Publishing the release is the last manual act. It starts the workflow `Release to Maven Central`, which does the rest.

10. Wait for the workflow `Release to Maven Central` to finish green:

    ```
    gh run list --workflow release-central.yml --limit 1
    gh run watch <run-id> --exit-status
    ```

    It builds from the tag, signs, uploads the bundle, lets the Central Portal publish it, waits for the files on `repo1.maven.org` and verifies them (the section "The publish run" below has the list). There is nothing to press in the portal. A green run means that `ai.humem:tiervec:<version>` resolves from Maven Central, which is the done-when of issue [#62](https://github.com/humemai/tiervec/issues/62): the first green run comments on that issue and closes it. A red run is read in its log; "The publish run" says what each failure means.
11. In a second pull request, set `<version>` on `main` to the next `-SNAPSHOT` (see above).

A tag is never moved or deleted. A mistake in a release is fixed by the next PATCH release, and the changelog says so. A version on Maven Central cannot be changed or removed once it is published.

## Maven Central

The artifact is `ai.humem:tiervec`. The namespace `ai.humem` is verified on the Central Portal (<https://central.sonatype.com>), account `taewoon@humem.ai`. Central accepts the artifact only with a complete `pom.xml` (name, description, url, license, developers, scm), a sources jar, a javadoc jar and a GnuPG signature on every file. The profile `release` in `pom.xml` builds and signs all of that, and the plugin `central-publishing-maven-plugin` uploads one bundle. The profile does not publish by default: the property `central.autoPublish` is `false` in `pom.xml`, so a local build and the dry run stop at `VALIDATED` or before. Only the publish run of the workflow passes `-Dcentral.autoPublish=true -Dcentral.waitUntil=published`, and then the Portal publishes the deployment as soon as it is valid. Creating the GitHub release is the deliberate act that starts it, and a published version cannot be taken back.

Only the workflow `release-central.yml` runs the profile with real credentials. Nobody runs it from a laptop.

### Secrets

Four repository secrets of `humemai/tiervec` (Settings, Secrets and variables, Actions). The workflow reads nothing else.

| Name | What it holds |
| --- | --- |
| `CENTRAL_USERNAME` | The user name of the Central Portal user token |
| `CENTRAL_PASSWORD` | The password of the same token |
| `GPG_PRIVATE_KEY` | The armored private signing key |
| `GPG_PASSPHRASE` | The passphrase of that key |

Check that the four names exist (the values cannot be read, and the command shows names only):

```
gh api repos/humemai/tiervec/actions/secrets --jq '[.secrets[].name]'
```

The workflow never prints a value, never puts one on a command line and never writes one to a file. The `settings.xml` it creates has `${env.CENTRAL_USERNAME}` and `${env.CENTRAL_PASSWORD}` references, mode 600, in the runner temp directory, and it removes the file and the GnuPG home at the end.

### The signing key

Create a key (a second key is not needed if one exists), store it as secrets without printing it, and publish the public key. `<FPR>` is the fingerprint that `gpg --list-secret-keys --keyid-format long` shows.

```
gpg --quick-generate-key "HumemAI tiervec releases <taewoon@humem.ai>" ed25519 sign 3y
gpg --list-secret-keys --keyid-format long taewoon@humem.ai

gpg --armor --export-secret-keys <FPR> | gh secret set GPG_PRIVATE_KEY --repo humemai/tiervec
gh secret set GPG_PASSPHRASE --repo humemai/tiervec

gpg --keyserver hkps://keyserver.ubuntu.com --send-keys <FPR>
gpg --armor --export <FPR> > tiervec-release-key.asc
```

`gh secret set GPG_PASSPHRASE` asks for the value, so it does not reach the shell history. Upload `tiervec-release-key.asc` at <https://keys.openpgp.org/upload> and follow the link in the e-mail it sends, then delete the file. Central looks the public key up on these key servers when it validates a signature, so a deployment fails validation until the key is published.

The token pair comes from the portal (Account, Generate User Token). Store it the same way, each command asks for its value:

```
gh secret set CENTRAL_USERNAME --repo humemai/tiervec
gh secret set CENTRAL_PASSWORD --repo humemai/tiervec
```

### Dry run

```
gh workflow run release-central.yml --ref <branch or main> -f dry-run=true
gh run list --workflow release-central.yml --limit 1
gh run view <run-id> --log-failed
```

The dry run is the default of the manual trigger and runs on any branch (GitHub dispatches a workflow only when its file is on `main`). It also runs on every pull request that changes `pom.xml`, `src/main` or the workflow file, so a Javadoc error or a broken release profile shows up in the pull request and not at release time; that check is not required. It generates a throwaway signing key inside the job, so it needs no secret and has no Central credentials: it cannot publish. It exports that key and imports it into a second GnuPG home with the script `.github/scripts/import-signing-key.sh`, the same script the publish run uses, so the import from an environment variable and the signing with a passphrase are rehearsed. It builds with JDK 21 (`-Prelease -DskipTests`, with a `-SNAPSHOT` version written as its release form in the job's checkout, because the Central plugin writes a bundle only for a release version), signs, writes the Central bundle, and then checks the bundle: the jar, sources jar, javadoc jar and pom are all there with checksums and a valid signature, the sources jar and the javadoc jar have content, and the pom has the elements Central requires. It uploads the bundle as the run artifact `central-dry-run-bundle` (7 days).

The dry run also checks and tests the scripts of the publish run: `bash -n` and `shellcheck` on every script in `.github/scripts`, and `.github/scripts/test-central-scripts.sh`. That test runs `verify-central-release.sh` and `check-central-absent.sh` against a local server that stands in for `repo1.maven.org` and the key server, throwaway signing keys and a Maven stub, so it needs nothing from the network. Each case spoils one thing (a flipped byte, a missing `.asc`, a wrong checksum, a bad signature, a signature of another key, a key that cannot be fetched, a jar that is not the built jar, a version that never appears) and the script must fail with the message of that thing; the good set must pass. To run it on a laptop: `PYTHON_BIN="$(uv run --no-project python -c 'import sys; print(sys.executable)')" bash .github/scripts/test-central-scripts.sh`.

It does not prove that the real key, the passphrase and the token work, or that the key is on a key server. The publish run checks the first two early: it imports the key with the same script, signs a probe file with the passphrase, and stops before the build if that fails.

The dry run covers what changes with the code, such as a Javadoc error. Run it on the release branch before the tag (step 6).

### The publish run

The run starts when a GitHub release is published, or by hand with `-f dry-run=false` on the tag. Creating the release is the deliberate act: nothing else starts a publish. It refuses to run unless the ref is a tag that equals the version in `pom.xml` (a bare version, no `-SNAPSHOT`), the tagged commit is on `main`, the version is not on Maven Central yet, and all four secrets are present. "Not on Central yet" means that `https://repo1.maven.org/maven2/ai/humem/tiervec/<version>/` answers 404, so a re-run cannot push a duplicate. An answer that is neither 404 nor 200 stops the run too, because "cannot tell" must not mean "go on".

What the run does, in order:

1. Checks the tag, the pom version, that the commit is on `main`, that the version is not on Central, and the four secrets.
2. Imports the signing key with `.github/scripts/import-signing-key.sh` and signs a probe file, so a wrong passphrase fails before the build. It keeps the fingerprint of the key (a public value) for step 6.
3. Builds with the profile `release` and the fast tests, signs the jar, the sources jar, the javadoc jar and the pom, and uploads the bundle to the Central Portal. An expired Central token shows up here as an HTTP 401.
4. The Portal validates the bundle and, because the run passes `-Dcentral.autoPublish=true -Dcentral.waitUntil=published`, publishes it. The Central plugin waits until the deployment is `PUBLISHED`, for at most `central.waitMaxTime` (3600 seconds).
5. Waits for the sync from the Portal to `repo1.maven.org`: it polls `https://repo1.maven.org/maven2/ai/humem/tiervec/<version>/tiervec-<version>.pom` every 30 seconds, for at most 60 minutes, and fails with the last answer when the file never appears. (The search index of <https://central.sonatype.com> and `search.maven.org` is slower than the repository; the check uses `repo1.maven.org` only.)
6. Verifies the result with `.github/scripts/verify-central-release.sh`: it downloads the jar, the sources jar, the javadoc jar and the pom with their `.asc` and checksum files; checks md5, sha1 and sha256 (and sha512 where served) against the bytes; fetches the public key from `keyserver.ubuntu.com` by the fingerprint of step 2 and checks that every signature is valid and made by that key; compares the three jars byte for byte with the jars that step 3 built, so what is on Central is what was built; and resolves `ai.humem:tiervec:<version>` with `dependency:get` in an empty local Maven repository with an isolated settings file (no mirrors, Central only).
7. Writes the result in the job summary. The first version that passes also comments on issue [#62](https://github.com/humemai/tiervec/issues/62) and closes it. A later version has no issue to comment on (releases are tracked by milestone, and a milestone or a GitHub release takes no comment), so the job summary and the run are its record.

To see it: `gh run watch <run-id> --exit-status` follows the run, and the job summary has the table of what Central serves. The Portal shows the deployment read-only at <https://central.sonatype.com/publishing/deployments>; there is nothing to press there.

What a failure means:

- Red in step 3 with an HTTP 401: the Central token expired or is wrong. Rotate it (issue [#97](https://github.com/humemai/tiervec/issues/97), the reminder below) and run the workflow again on the same tag.
- Red in step 3 or 4 because the deployment is `FAILED`: nothing was published. Read the messages in the portal, drop the deployment, fix the cause, and run the workflow again on the same tag (`gh workflow run release-central.yml --ref <version> -f dry-run=false`). Fixing a problem in the sources needs a new version, because a tag is never moved.
- Red in step 5 or 6: the version is published and is not taken back, whatever the verification said. If the pom never appeared within 60 minutes, run the workflow `Verify on Maven Central` (below) later. If a checksum, a signature or a jar differs, that is a real defect of the release: the fix is the next patch version, and its changelog says so.
- A deployment that sits at `VALIDATED` (the run was stopped, or it was started with `auto-publish` set to `false`): press Publish for `ai.humem:tiervec:<version>` at <https://central.sonatype.com/publishing/deployments>, then run `Verify on Maven Central`. Do not run the publish run again while that deployment is in the portal: drop it first, because the check "not on Central yet" cannot see a deployment that is not synced.
- To choose the manual path from the start, run the workflow by hand on the tag with `-f dry-run=false -f auto-publish=false`. It uploads, waits for `VALIDATED` and stops without publishing, the run summary says so, and the steps above for a deployment at `VALIDATED` apply. A release event always auto-publishes.

### Verify on Maven Central

The workflow `Verify on Maven Central` (`verify-central.yml`) runs the same verification for a version that is already out, with no secret and no write permission:

```
gh workflow run verify-central.yml -f version=<version> -f fingerprint=<FPR>
```

`<FPR>` is the 40 hex digits of the signing key: the line `signing key fingerprint:` in the log of the publish run, or `gpg --list-keys --with-colons` for the key on a laptop. It cannot compare the jars with a build, because it has none. Use it after a manual Publish, after a publish run that timed out while waiting, and to check an old version again.

### Reminder: the Central token expires

The Central user token was created on 2026-11-12 with an expiry of one year, so it expires around 2027-11. Rotate it before then: generate a new token in the portal, set `CENTRAL_USERNAME` and `CENTRAL_PASSWORD` again, and keep the date of the new expiry here. An expired token shows as an authentication error (HTTP 401) in the upload step of the publish run, and the dry run cannot catch it because it uses no token. If the signing key has an expiry (the example above has three years), that date is a second rotation point: extend it with `gpg --edit-key <FPR> expire`, export it again as above and upload the public key again.
