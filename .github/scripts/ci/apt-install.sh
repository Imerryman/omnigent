#!/usr/bin/env bash
# Installs apt packages on a GitHub-hosted runner, with a bounded retry.
#
# Usage:  apt-install.sh [--with-recommends] PACKAGE...
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

set -euo pipefail

recommends=(--no-install-recommends)
if [[ "${1:-}" == "--with-recommends" ]]; then
  recommends=()
  shift
fi

if [[ $# -eq 0 ]]; then
  echo "apt-install.sh: no packages given" >&2
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive

attempts=3
for attempt in $(seq 1 "$attempts"); do
  if sudo -E apt-get update -qq \
    && sudo -E apt-get install -y -q "${recommends[@]}" "$@"; then
    exit 0
  fi
  if [[ "$attempt" -lt "$attempts" ]]; then
    echo "apt-install.sh: attempt $attempt/$attempts failed for [$*]; retrying" >&2
    sleep $((attempt * 5))
  fi
done

echo "apt-install.sh: giving up after $attempts attempts for [$*]" >&2
exit 1
