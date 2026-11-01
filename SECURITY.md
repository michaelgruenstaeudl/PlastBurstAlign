# Security Policy

## Supported versions

Security fixes are provided for the latest release of the current major version
(3.x, including release candidates until 3.0.0 is out). Older versions are not
patched; please upgrade.

| Version | Supported |
|---|---|
| 3.x | yes |
| < 3.0 | no |

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately in one of these ways:

1. GitHub: open the repository's **Security** tab and choose
   **Report a vulnerability** (private vulnerability reporting), or
2. e-mail the maintainer, Michael Gruenstaeudl, at
   [m_gruenstaeudl@fhsu.edu](mailto:m_gruenstaeudl@fhsu.edu) with the subject
   `PlastBurstAlign security`.

Please include the affected version (`plastburstalign --doctor` output helps),
a description of the impact, and the smallest input or command that reproduces
the problem. We aim to acknowledge a report within 7 days, to keep you informed
while we investigate, and to coordinate a fix and a disclosure date with you. We
credit reporters in the changelog unless they prefer to stay anonymous.

## Scope and trust model

PlastBurstAlign is a batch tool that is normally run by a user on their own
data. Things worth knowing:

* **GenBank input is parsed, not executed.** Malformed or hostile flat files may
  cause errors, which the tool isolates per record/feature/locus; a crash,
  hang or unbounded memory use triggered by a crafted file is in scope.
* **External programs are executed.** The aligners (MAFFT, MUSCLE, Clustal
  Omega) are started as subprocesses without a shell. The `--alignmentcommand`
  option and the YAML `--config` file are the exception by design: they take a
  command template that is run through a shell, so only use configuration and
  command lines that you trust. Treat config files from untrusted sources like
  scripts.
* **Scratch and output directories** are created with the user's default
  permissions; on shared systems choose a private `--scratch-dir`/`--outd`.
* **PlastBench dataset downloads** (`benchmarking/plastbench/plastbench.py fetch`)
  verify a SHA-256 (or the Zenodo MD5) before extraction and reject archives
  containing absolute paths, `..` components or links. Do not bypass this with
  `--allow-unverified` for archives you do not trust.
* Releases are published to PyPI with trusted publishing and the artifacts are
  signed with Sigstore (see `.github/workflows/build-and-publish.yaml`).
