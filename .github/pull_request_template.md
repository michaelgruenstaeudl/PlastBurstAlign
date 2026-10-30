## Summary

<!-- What does this change and why? Link related issues (Fixes #123). -->

## Type of change

- [ ] Bug fix
- [ ] New feature (backwards compatible)
- [ ] Performance / memory improvement
- [ ] Breaking change (explain the migration path below)
- [ ] Documentation, CI or packaging only

## Checklist

- [ ] Tests added or updated (tiny, offline, unittest-style) and `python -m pytest -q -p no:cacheprovider tests` passes
- [ ] `ruff check .` and `ruff format --check .` are clean
- [ ] `mypy plastburstalign` is clean
- [ ] Golden parity gate passes: `python benchmarking/golden_parity.py check` (or the pull request explains why outputs legitimately change and updates `tests/golden/`)
- [ ] `CHANGELOG.md` has an entry under **Unreleased**
- [ ] Documentation / docstrings updated for user-visible changes
- [ ] Performance or memory claims are backed by a PlastBench result (attach `results.json` or the `report` table)
- [ ] Commit messages follow Conventional Commits (`type(scope): summary`)

## Testing notes

<!-- Which aligners, Python versions and operating systems did you try? Output of `plastburstalign --doctor` is helpful for scheduling/hardware changes. -->
