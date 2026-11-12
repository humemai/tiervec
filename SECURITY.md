# Security policy

## Reporting a vulnerability

Do not open a public issue for a security problem.

Use GitHub's private reporting: open the [Security tab](https://github.com/humemai/tiervec/security) of the repository and choose **Report a vulnerability**. Or send an email to [info@humem.ai](mailto:info@humem.ai) with the subject `tiervec security`. Either way a maintainer reads it, replies, and works with you on a fix before anything is made public.

Please include:

- The version or the commit of tiervec.
- The JDK (the output of `java -version`), the operating system and the CPU architecture.
- Whether the Vector API module is on (`--add-modules jdk.incubator.vector`), or the scalar twins are in use, or the code runs in a native image.
- A description of the problem and its impact.
- Steps to reproduce it, or a small program that shows it.

## What counts

tiervec is a library that runs in the process of its host. It opens no socket and reads no file that the host did not point it to. These are in scope:

- A file in a segment directory that makes `TieredIndex.open` or `SegmentReader` read or write memory outside the file, loop without end, or allocate without a bound, other than by throwing an exception. The format checks its sections with CRC32C, but a CRC detects accidental damage and is no protection against tampering (see [docs/SEGMENT-FORMAT.md](docs/SEGMENT-FORMAT.md)), so a hostile file must still not do harm.
- A search that returns a vector for an id that was deleted, or that returns data from another index.
- A flaw in the build or in the workflow files that lets an outside party run code in CI or change a release.

A wrong answer, a crash on bad input that ends in an exception, and a slow query are ordinary bugs. Open an issue for them.

## Supported versions

The first release is `0.1.0`. A fix goes to the latest minor version as a new patch version, and `main` gets it too (see the version rule in [docs/RELEASING.md](docs/RELEASING.md)).

## What to expect

tiervec is a small open source project with one maintainer and no fixed response time. You get a reply, and credit in the fix if you want it. No bounty is paid.
