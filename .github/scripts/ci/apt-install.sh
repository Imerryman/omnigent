#!/bin/sh
# Installs apt packages on a GitHub-hosted runner, with a bounded retry.
#
# Usage:  apt-install.sh [--with-recommends] PACKAGE...
# Exits:  0 installed, 1 still failing after 3 attempts, 2 no packages given.
#
# Why this exists: the runner images ship a PRE-SEEDED apt index. When the
# upstream Ubuntu mirror rotates a package to a newer version, that stale index
# still points at a .deb that no longer exists, and the install dies with
#     E: Failed to fetch .../bubblewrap_0.9.0-1ubuntu0.1_amd64.deb  404  Not Found
# So `apt-get update` is mandatory, not optional, before every install. The
# retry then absorbs a transient mirror hiccup (a mirror mid-rotation serves a
# 404 for a few seconds) instead of reddening the whole matrix.
#
# --with-recommends keeps apt's default Recommends handling. Without it we pass
# --no-install-recommends, which is a no-op for the tools we install here
# (bubblewrap's only Recommends is procps, already on the runner image) but is
# NOT verified for every package, hence the opt-out.
#
# POSIX sh, no bashisms: it is invoked as `bash <script>` from workflow steps
# today, but nothing here needs bash and /bin/sh keeps it portable.
#
# Deliberately NOT passing `-o APT::Update::Error-Mode=any` to apt-get update:
# it would turn a transient failure on any third-party index we do not even
# need (docker, microsoft, the git-core PPA -- all present on the runner image)
# into a hard job failure after the retries. Note this is an AVAILABILITY
# TRADEOFF, not a strict improvement: a stale index that still references a
# downloadable .deb installs fine without being refreshed, so tolerating index
# errors can mask a refresh failure. We take that trade because THE OBSERVED
# failure -- a 404 on the package fetch itself -- surfaces as an install
# failure, which this loop retries, re-running update each time.
#
# Not every caller can use this file. Seven jobs do; the other seven install
# inline because the tree they check out may not contain this script: four
# check out an arbitrary historical revision (flake-stress*.yml,
# benchmark.yml), and three pin `ref: default_branch` while their own workflow
# definition comes from the triggering ref (issue-triage.yml,
# security-triage.yml, polly-review.yml). See the comments at those sites.

set -eu

no_recommends=1
if [ "${1:-}" = "--with-recommends" ]; then
  no_recommends=
  shift
fi

if [ "$#" -eq 0 ]; then
  echo "apt-install.sh: no packages given" >&2
  exit 2
fi

packages="$*"
if [ -n "$no_recommends" ]; then
  set -- --no-install-recommends "$@"
fi

export DEBIAN_FRONTEND=noninteractive

# 3 attempts, 5s then 10s of backoff. Keep in step with the inline copies.
for attempt in 1 2 3; do
  if sudo -E apt-get update -qq \
    && sudo -E apt-get install -y -q "$@"; then
    exit 0
  fi
  if [ "$attempt" -lt 3 ]; then
    echo "apt-install.sh: attempt $attempt/3 failed for [$packages]; retrying" >&2
    sleep $((attempt * 5))
  fi
done

echo "apt-install.sh: giving up after 3 attempts for [$packages]" >&2
exit 1
