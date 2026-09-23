#!/usr/bin/env python3
"""Regression check for the apt install retry logic, against a MOCKED apt.

Run it by hand after touching any apt install in .github/:

    python3 .github/scripts/ci/apt-install-selftest.py

There are two implementations of the same retry loop and they must not drift:

  * .github/scripts/ci/apt-install.sh -- used by the jobs that run with the
    workflow's own revision checked out.
  * an inline copy in each job whose checked-out tree may not contain that
    script (see the "INLINE ON PURPOSE" comment at each such call site).

This script finds every inline copy by that marker -- so a new one is picked up
automatically -- and asserts all of them behave identically: 3 attempts, 5s then
10s of backoff, exit 1 once exhausted, exit 0 as soon as an attempt succeeds,
and execution continuing past the loop on success. The helper is additionally
checked under bash, sh and dash for its exit-2 (no packages) path and its
--with-recommends escape hatch.

No apt, sudo or network is involved: PATH is pointed at stubs that record calls
and fail on demand.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
HELPER = ROOT / ".github/scripts/ci/apt-install.sh"
MARKER = "INLINE ON PURPOSE"

STUBS = {
    # Drops the -E and execs the rest, so `sudo -E apt-get ...` reaches apt-get.
    "sudo": '#!/bin/sh\n[ "$1" = "-E" ] && shift\nexec "$@"\n',
    # Fails every `install` until MOCK_SUCCEED_ON; `update` always succeeds.
    "apt-get": (
        '#!/bin/sh\n'
        'echo "apt-get $*" >> "$MOCK_LOG"\n'
        'case "$*" in *install*)\n'
        '  n=$(( $(cat "$MOCK_N" 2>/dev/null || echo 0) + 1 )); echo $n > "$MOCK_N"\n'
        '  [ "$n" -ge "${MOCK_SUCCEED_ON:-999}" ] && exit 0\n'
        '  echo "E: Failed to fetch ... 404  Not Found" >&2; exit 100 ;;\n'
        'esac\nexit 0\n'
    ),
    "sleep": '#!/bin/sh\necho "sleep $*" >> "$MOCK_LOG"\n',
}
# Commands that follow the install inside the same `run:` block. Stubbing them
# lets us assert the block CONTINUES past a successful install.
for _cmd in ("sysctl", "pnpm", "node", "uv", "npm"):
    STUBS[_cmd] = '#!/bin/sh\necho "%s $*" >> "$MOCK_LOG"\nexit 0\n' % _cmd

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail and not ok else ''}")
    if not ok:
        failures.append(name)


def run(script: str, args: list[str], shell: str, succeed_on: int, bindir: str, log: pathlib.Path):
    log.write_text("")
    n = log.parent / "n"
    n.unlink(missing_ok=True)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "MOCK_LOG": str(log),
        "MOCK_N": str(n),
        "MOCK_SUCCEED_ON": str(succeed_on),
        # The inline blocks run under `shell: bash` with these defaults.
        "DEBIAN_FRONTEND": "",
        # Always set on a real runner; some blocks append to them after the
        # install, and `set -u` would abort on an unset one.
        "GITHUB_WORKSPACE": str(ROOT),
        "GITHUB_PATH": str(log.parent / "github_path"),
        "GITHUB_ENV": str(log.parent / "github_env"),
        "GITHUB_OUTPUT": str(log.parent / "github_output"),
    }
    src = script if shell != "inline" else "set -euo pipefail\n" + script
    proc = subprocess.run(
        ["bash" if shell == "inline" else shell, "-c", src, "_"] + args,
        capture_output=True, text=True, env=env, cwd=ROOT,
    )
    return proc.returncode, log.read_text()


def inline_blocks() -> list[tuple[str, str]]:
    found = []
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        doc = yaml.safe_load(path.read_text()) or {}
        for job_name, job in (doc.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                body = step.get("run") or ""
                if MARKER in body:
                    found.append((f"{path.name}:{job_name}", body))
    return found


def main() -> int:
    helper_src = HELPER.read_text()
    blocks = inline_blocks()

    with tempfile.TemporaryDirectory() as tmp:
        bindir = pathlib.Path(tmp) / "bin"
        bindir.mkdir()
        for name, body in STUBS.items():
            p = bindir / name
            p.write_text(body)
            p.chmod(0o755)
        log = pathlib.Path(tmp) / "log"
        b, L = str(bindir), log

        print(f"helper: {HELPER.relative_to(ROOT)}")
        for sh in ("bash", "sh", "dash"):
            if shutil.which(sh) is None:
                print(f"  SKIP  {sh} not installed")
                continue
            rc, out = run(helper_src, ["bubblewrap", "tmux"], sh, 999, b, L)
            check(f"{sh}: persistent failure -> 1", rc == 1, f"rc={rc}")
            check(f"{sh}: 3 attempts, 5s+10s backoff",
                  out.count("apt-get install") == 3 and "sleep 5" in out and "sleep 10" in out, out)
            rc, _ = run(helper_src, ["bubblewrap", "tmux"], sh, 3, b, L)
            check(f"{sh}: recovers on 3rd attempt -> 0", rc == 0, f"rc={rc}")
            rc, _ = run(helper_src, [], sh, 1, b, L)
            check(f"{sh}: no packages -> 2", rc == 2, f"rc={rc}")
            rc, _ = run(helper_src, ["--with-recommends"], sh, 1, b, L)
            check(f"{sh}: --with-recommends with no packages -> 2", rc == 2, f"rc={rc}")
            rc, out = run(helper_src, ["bubblewrap"], sh, 1, b, L)
            check(f"{sh}: default passes --no-install-recommends",
                  rc == 0 and "--no-install-recommends bubblewrap" in out, out)
            rc, out = run(helper_src, ["--with-recommends", "libmysqlclient-dev"], sh, 1, b, L)
            check(f"{sh}: --with-recommends restores apt defaults",
                  rc == 0 and "--no-install-recommends" not in out, out)

        print(f"\ninline copies found via '{MARKER}': {len(blocks)}")
        if not blocks:
            failures.append("no inline blocks found -- marker renamed?")
        for where, body in blocks:
            print(f"{where}")
            rc, out = run(body, [], "inline", 999, b, L)
            check("persistent failure -> 1", rc == 1, f"rc={rc}")
            check("3 attempts, 5s+10s backoff",
                  out.count("apt-get install") == 3 and "sleep 5" in out and "sleep 10" in out, out)
            rc, out = run(body, [], "inline", 3, b, L)
            check("recovers on 3rd attempt -> 0 and continues past the loop", rc == 0, f"rc={rc}\n{out}")
            rc, out = run(body, [], "inline", 1, b, L)
            recs = "--no-install-recommends" in out
            want = "libmysqlclient-dev" not in body
            check(f"recommends flag {'present' if want else 'absent'}", recs is want, out)

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + "; ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
