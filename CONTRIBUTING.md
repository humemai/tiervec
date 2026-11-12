# Contributing to tiervec

tiervec is a Java library of vector-index algorithms that is meant to run inside a database engine. It is experimental: there is no release and the API is not stable yet (see [docs/STATUS.md](docs/STATUS.md)). Small, focused contributions are welcome.

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

The fast tests run on every pull request, on JDK 21 and 25. The slow tests also run on every pull request, on x86-64 Linux, and take about 45 minutes. You do not have to run them on your own machine.

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

Three checks are required before a merge:

- `fast tests (JDK 21)`
- `fast tests (JDK 25)`
- `Ensure SHA pinned actions`

The other jobs (arm64, Windows, native image, slow tests) also run on a pull request. A failure there is read like any other failure, even where the check is not required.

### Workflow files

Every action in `.github/workflows/` is pinned to a full commit SHA, with the version as a comment. A floating tag fails the check `Ensure SHA pinned actions`.

### Measurements

Do not put a performance number in a pull request or in the documentation unless you can say how it was measured: the hardware, the JDK, the data, the number of runs. Timing numbers from the shared GitHub runners are useful as ratios between two things measured in the same run. They are not absolute speed.

## Conduct

Be polite and be specific. Disagree with the code or the idea, not with the person. Harassment is not tolerated, and the maintainer may remove comments and block people who harass others.
