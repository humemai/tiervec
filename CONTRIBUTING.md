# Contributing to tiervec

tiervec is a Java library of vector-index algorithms that is meant to run inside a database engine. It is experimental: the first release is `0.1.0` and the API is not stable yet (see [docs/STATUS.md](docs/STATUS.md)). Small, focused contributions are welcome.

## Licence

tiervec is licensed under the [Apache License 2.0](LICENSE). When you submit a contribution, you agree that it is licensed under the same terms (Section 5 of the licence). You keep your copyright. There is no separate contributor agreement.

Do not copy code or text into the repository unless its licence is compatible with Apache 2.0. Papers and libraries that the work builds on are listed in [THIRD_PARTY.md](THIRD_PARTY.md). The library has no run-time dependencies, and a new one needs a strong reason.

## Before you start

- A security problem is not an issue. Follow [SECURITY.md](SECURITY.md).
- A question is not a bug. See [SUPPORT.md](SUPPORT.md).
- Search the [open issues](https://github.com/humemai/tiervec/issues) first. If your idea is not there, open an issue and say what problem it solves. For a new feature or a change in behavior, wait for an answer before you write much code.
- Read [docs/DESIGN.md](docs/DESIGN.md) and [docs/PROVIDER-CONTRACT.md](docs/PROVIDER-CONTRACT.md). They say what the library does and what it promises to a host database.

## Build and test

You need JDK 21 or newer and Maven 3.9 or newer.

```
mvn -B -ntp verify            # fast tests: Vector API kernels, then the scalar twins
mvn -B -ntp verify -Pslow     # the long concurrency and stress tests (tag "slow")
```

The fast tests run on every pull request, on JDK 21 and 25. The slow tests take about 40 minutes per JDK, so they do not run on an ordinary pull request: see [What CI runs, and when](#what-ci-runs-and-when). You do not have to run them on your own machine.

### Build the documentation

The site at <https://docs.humem.ai/tiervec/dev/> is built with MkDocs Material from the README, the other Markdown files in the repository root, and `docs/`. The hook `docs-site/hooks.py` turns the root files into pages and rewrites relative links: a link to another page stays a link to that page, and a link to any other file in the repository becomes a link to it on GitHub. The design system in `docs/brand` is a vendored copy. Do not edit it: change it in [humemai/design-system](https://github.com/humemai/design-system) and vendor it again. `bash docs/brand/verify.sh` checks the copy. To build the site as the check does:

```
uv run --no-project --python 3.12 --with-requirements docs-site/requirements.txt mkdocs build --strict
```

The build fails on a link that points nowhere. The workflow `deploy-docs.yml` publishes the site as `dev` when a change to the docs reaches `main`.

## Make a change

1. Fork the repository and branch from `main`.
2. Keep the change small. One pull request does one thing.
3. Add a test. A test for a bug fix must fail on the old code and pass on the new. Say in the pull request how you checked that.
4. Run the fast tests. A change in a kernel must keep the scalar twin and the Vector API form in agreement (see `KernDiffTest`).
5. Update the documentation that your change makes wrong: the README, the files in `docs/`, and the Javadoc.
6. Open a pull request. Say what changed and why, and write `Closes #N` for the issue it fixes.

Commit messages start with the area, then a short summary and the issue number, for example `Tests: replace absolute throughput floors in concurrency tests with an overlap proof (#59)`.

Pull requests are squash-merged into `main`, so the history is linear.

### Checks that must pass

Four checks are required before a merge:

- `fast tests (JDK 21)`
- `fast tests (JDK 25)`
- `Ensure SHA pinned actions`
- `docs build (strict)`

The other jobs of a pull request (arm64, Windows, the benchmark smoke, native image) are not required. A failure there is read like any other failure.

### What CI runs, and when

| When | What runs |
| --- | --- |
| Every pull request | The fast tests (x86-64 Linux on JDK 21 and 25, arm64 Linux, arm64 macOS, Windows x64), the benchmark smoke, the documentation build and the workflow check. The native image jobs and the Maven Central dry run start only when the files they cover change. It ends in about 12 minutes. |
| A pull request with the label `ci:slow` | The same, the slow tests on JDK 21 and 25 (about 40 minutes each), and the two slow crash tests on Windows x64. |
| Every push to `main` | Everything above, the slow tests included. |
| Every day at 19:17 UTC | The slow tests, on the head of `main`. |

To ask for the slow tests on a pull request, add the label `ci:slow`. If you cannot set labels, ask for it in the pull request and the maintainer adds it. The slow tests start when the label is added and run again on every later push to that pull request. Ask for them when a change touches what they cover: concurrent readers and writers, seals and merges, record-id reuse, or the batch paths. On a pull request without the label, the slow tests show as one skipped check, which is not a failure.

The slow tests also run by hand on any branch: `gh workflow run slow-tests.yml --ref <branch>`.

When a slow run on `main` is red, the workflow opens the issue "Slow tests are red on main", or comments on it when it is already open. The slow tests are in `.github/workflows/slow-tests.yml`; the other tests are in `ci.yml`.

### Workflow files

Every action in `.github/workflows/` is pinned to a full commit SHA, with the version as a comment. A floating tag fails the check `Ensure SHA pinned actions`.

### Measurements

Do not put a performance number in a pull request or in the documentation unless you can say how it was measured: the hardware, the JDK, the data, the number of runs. Timing numbers from the shared GitHub runners are useful as ratios between two things measured in the same run. They are not absolute speed.

## Conduct

Be polite and be specific. Disagree with the code or the idea, not with the person. Harassment is not tolerated, and the maintainer may remove comments and block people who harass others.
