# Releasing

This page is the rule for version numbers and the steps of a release. There is no release yet: the version on `main` is `0.1.0-SNAPSHOT`.

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
- The artifact `ai.humem:tiervec:<version>` on Maven Central: the jar, a sources jar and a javadoc jar, each signed. The GitHub release starts the upload (workflow `release-central.yml`) and the maintainer confirms it in the Central Portal. The details are in the section "Maven Central" below ([#62](https://github.com/humemai/tiervec/issues/62)).

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

   Publishing the release starts the workflow `Release to Maven Central`. It builds from the tag, runs the fast tests, signs, uploads the bundle and waits until the portal has validated it. It does not publish.

10. Confirm the upload in the Central Portal and press Publish (see "Maven Central" below). Then check that a small project resolves `ai.humem:tiervec:<version>` from Maven Central. That is the done-when of issue [#62](https://github.com/humemai/tiervec/issues/62).
11. In a second pull request, set `<version>` on `main` to the next `-SNAPSHOT` (see above).

A tag is never moved or deleted. A mistake in a release is fixed by the next PATCH release, and the changelog says so. A version on Maven Central cannot be changed or removed once it is published.

## Maven Central

The artifact is `ai.humem:tiervec`. The namespace `ai.humem` is verified on the Central Portal (<https://central.sonatype.com>), account `taewoon@humem.ai`. Central accepts the artifact only with a complete `pom.xml` (name, description, url, license, developers, scm), a sources jar, a javadoc jar and a GnuPG signature on every file. The profile `release` in `pom.xml` builds and signs all of that, and the plugin `central-publishing-maven-plugin` uploads one bundle. The profile has `autoPublish` set to `false`: the deployment stops at `VALIDATED` and nothing becomes public until the maintainer presses Publish in the portal.

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

The dry run is the default of the manual trigger and runs on any branch (GitHub dispatches a workflow only when its file is on `main`). It also runs on every pull request that changes `pom.xml`, `src/main` or the workflow file, so a Javadoc error or a broken release profile shows up in the pull request and not at release time; that check is not required. It generates a throwaway signing key inside the job, so it needs no secret and has no Central credentials: it cannot publish. It builds with JDK 21 (`-Prelease -DskipTests`), signs, writes the Central bundle, and then checks the bundle: the jar, sources jar, javadoc jar and pom are all there with checksums and a valid signature, the sources jar and the javadoc jar have content, and the pom has the elements Central requires. It uploads the bundle as the run artifact `central-dry-run-bundle` (7 days).

It does not prove that the real key, the passphrase and the token work, or that the key is on a key server. The publish run checks the first two early: it imports the key, signs a probe file with the passphrase, and stops before the build if that fails.

The dry run covers what changes with the code, such as a Javadoc error. Run it on the release branch before the tag (step 6).

### The publish run

The run starts when a GitHub release is published, or by hand with `-f dry-run=false` on the tag. It refuses to run unless the ref is a tag that equals the version in `pom.xml` (a bare version, no `-SNAPSHOT`), the tagged commit is on `main`, and all four secrets are present.

To confirm in the portal:

1. Open <https://central.sonatype.com/publishing/deployments>.
2. Find the deployment named `ai.humem:tiervec:<version>`. Its status must be `VALIDATED`.
3. Press Publish. The status goes through `PUBLISHING` to `PUBLISHED`, and the files appear under <https://repo.maven.apache.org/maven2/ai/humem/tiervec/> after a while (the search index is slower than the repository).

If the status is `FAILED`, read the messages in the portal, drop the deployment, fix the cause, and run the workflow again on the same tag. That is allowed because nothing was published. Fixing a problem in the sources needs a new version.

### Reminder: the Central token expires

The Central user token was created on 2026-11-12 with an expiry of one year, so it expires around 2027-11. Rotate it before then: generate a new token in the portal, set `CENTRAL_USERNAME` and `CENTRAL_PASSWORD` again, and keep the date of the new expiry here. An expired token shows as an authentication error (HTTP 401) in the upload step of the publish run, and the dry run cannot catch it because it uses no token. If the signing key has an expiry (the example above has three years), that date is a second rotation point: extend it with `gpg --edit-key <FPR> expire`, export it again as above and upload the public key again.

## Before the first release

These open points are named in the repository and are settled before `0.1.0` is tagged:

- Settled: the Java package is `ai.humem.tiervec` and the Maven coordinates are `ai.humem:tiervec`. The package was renamed before the first release, so the rename is not a breaking change. After the release a rename would be one.
- Comments and Javadoc in the sources still use names from the first experiment ([STATUS.md](STATUS.md#import-notes)).
- The supported surface is listed in [API.md](API.md).
- Settled: the documentation site publishes only the label `dev` ([docs.humem.ai/tiervec/dev](https://docs.humem.ai/tiervec/dev/)) until the first release. `deploy-docs.yml` has a tag trigger that publishes the release version with the alias `latest`, and `dev` stays as the label of `main`.
- Settled: the workflow `release-central.yml` and the profile `release` exist ([#62](https://github.com/humemai/tiervec/issues/62)). Before the tag, the four secrets of "Maven Central" must exist and the dry run must be green on the release branch.
