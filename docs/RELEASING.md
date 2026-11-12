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

No jar is published yet. Publishing to Maven Central is a separate step with its own issue ([#62](https://github.com/humemai/tiervec/issues/62)).

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
6. Open a pull request named `Release <version>`, wait for green, and merge it.
7. Tag the merge commit and push the tag. The tag is the bare version:

   ```
   git tag <version> <merge-sha>
   git push origin <version>
   ```

8. Publish the GitHub release. The notes are the changelog section of this version:

   ```
   gh release create <version> --title <version> --notes-file <file with the section>
   ```

9. In a second pull request, set `<version>` on `main` to the next `-SNAPSHOT` (see above).

A tag is never moved or deleted. A mistake in a release is fixed by the next PATCH release, and the changelog says so.

## Before the first release

These open points are named in the repository and are settled before `0.1.0` is tagged:

- Settled: the Java package is `ai.humem.tiervec` and the Maven coordinates are `ai.humem:tiervec`. The package was renamed before the first release, so the rename is not a breaking change. After the release a rename would be one.
- Settled: comments and Javadoc in the sources use plain wording. The requirement ids, fix numbers and rule names of the first experiment are gone ([STATUS.md](STATUS.md#import-notes)).
- The supported surface is listed in [API.md](API.md).
- The documentation site publishes only the label `dev` ([docs.humem.ai/tiervec/dev](https://docs.humem.ai/tiervec/dev/)), because there is no release and no `latest` alias. The release pull request gives `deploy-docs.yml` a tag trigger that publishes the release version with the alias `latest`, and `dev` stays as the label of `main`.
