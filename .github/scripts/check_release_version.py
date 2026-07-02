"""Fail a release build when the git tag and the built package version disagree.

Run in CI after ``python -m build``. Compares the release tag (e.g. ``v0.2.0``,
passed as ``argv[1]`` from ``github.ref_name``) against the version of the wheel
that was actually built into ``dist/``. Uses ``packaging`` so that normalised
equivalents match (e.g. tag ``v0.1`` == built ``0.1.0``, ``v0.2.0a`` ==
``0.2.0a0``).

This guards against the failure mode where a ``vX.Y.Z`` release is cut without
bumping ``version`` in ``pyproject.toml``: the build then produces an
already-published ``X.Y.(Z-1)`` artifact and PyPI rejects it with an opaque
``400 File already exists`` only after the build succeeds. Here we fail fast,
before the publish job runs, with an actionable message.

Usage::

    python .github/scripts/check_release_version.py "$TAG" [dist_dir]
"""

import sys
from pathlib import Path

from packaging.utils import parse_wheel_filename
from packaging.version import Version


def main(argv: list[str]) -> int:
  if len(argv) < 2:
    print(
        '::error::usage: check_release_version.py <tag> [dist_dir]',
        file=sys.stderr,
    )
    return 2

  tag = argv[1]
  # Release tags are conventionally `vX.Y.Z`; the package version has no `v`.
  expected = tag[1:] if tag[:1] in ('v', 'V') else tag
  dist_dir = argv[2] if len(argv) > 2 else 'dist'

  wheels = sorted(Path(dist_dir).glob('*.whl'))
  if not wheels:
    print(
        f'::error::no wheel found in {dist_dir}/ -- did `python -m build` run?',
        file=sys.stderr,
    )
    return 2

  # parse_wheel_filename returns (name, Version, build_tag, tags).
  built = parse_wheel_filename(wheels[0].name)[1]

  try:
    expected_version = Version(expected)
  except Exception:  # noqa: BLE001 - surface a clear CI error, not a traceback.
    print(
        f'::error::release tag {tag!r} does not contain a PEP 440 version '
        f'({expected!r}).',
        file=sys.stderr,
    )
    return 1

  if expected_version != built:
    print(
        f'::error::Release tag {tag!r} (version {expected_version}) does not '
        f'match the built package version {built}. Bump `version` in '
        f'pyproject.toml to {expected_version} (or fix the tag) before '
        f'publishing.',
        file=sys.stderr,
    )
    return 1

  print(f'OK: release tag {tag} matches built package version {built}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main(sys.argv))
