#!/usr/bin/env python3
"""Regression check for the apt install retry logic, against a MOCKED apt.

Run it by hand after touching any apt install in .github/:

    python3 .github/scripts/ci/apt-install-selftest.py

There are two implementations of the same retry loop and they must not drift:

  * .github/scripts/ci/apt-install.sh -- used by the jobs that run with the
    workflow's own revision checked out.
  * an inline copy in each job whose checked-out tree may not contain that
    script (see the "INLINE ON PURPOSE" comment at each such call site).

This script finds every inline copy by that marker and asserts all of them
behave identically: 3 attempts, 5s then 10s of backoff, exit 1 once exhausted,
exit 0 as soon as an attempt succeeds WITH the rest of the step still running,
and a failing `apt-get update` retried the same way as a failing install. The
helper is additionally checked under bash, sh and dash for its exit-2 (no
packages) path and its --with-recommends escape hatch.

Discovery is cross-checked: every `run:` block in .github/ that calls apt-get
install directly MUST carry the marker, so deleting a marker fails the run
rather than silently shrinking the check count.

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
SENTINEL = "SENTINEL_REACHED_END_OF_STEP"

STUBS = {
    # Drops the -E and execs the rest, so `sudo -E apt-get ...` reaches apt-get.
    "sudo": '#!/bin/sh\n[ "$1" = "-E" ] && shift\nexec "$@"\n',
    # `update` fails while its call count is below MOCK_UPDATE_OK_ON; `install`
    # fails while its own count is below MOCK_SUCCEED_ON. Separate counters, so
    # an update-only failure can be distinguished from an install-only one.
    "apt-get": (
        "#!/bin/sh\n"
        'echo "apt-get $*" >> "$MOCK_LOG"\n'
        'case "$*" in\n'
        "  *install*)\n"
        '    n=$(( $(cat "$MOCK_N" 2>/dev/null || echo 0) + 1 )); echo $n > "$MOCK_N"\n'
        '    [ "$n" -ge "${MOCK_SUCCEED_ON:-999}" ] && exit 0\n'
        '    echo "E: Failed to fetch ...deb  404  Not Found" >&2; exit 100 ;;\n'
        "  *update*)\n"
        '    u=$(( $(cat "$MOCK_U" 2>/dev/null || echo 0) + 1 )); echo $u > "$MOCK_U"\n'
        '    [ "$u" -ge "${MOCK_UPDATE_OK_ON:-1}" ] && exit 0\n'
        '    echo "E: Failed to fetch ...InRelease  Connection failed" >&2; exit 100 ;;\n'
        "esac\nexit 0\n"
    ),
    "sleep": '#!/bin/sh\necho "sleep $*" >> "$MOCK_LOG"\n',
}
# Commands that follow the install inside the same `run:` block. Stubbing them
# lets us assert the block CONTINUES past a successful install.
for _cmd in ("sysctl", "pnpm", "node", "uv", "npm"):
    STUBS[_cmd] = f'#!/bin/sh\necho "{_cmd} $*" >> "$MOCK_LOG"\nexit 0\n'

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    suffix = f"  -- {detail}" if detail and not ok else ""
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{suffix}")
    if not ok:
        failures.append(name)


def run(
    script: str,
    args: list[str],
    shell: str,
    bindir: str,
    log: pathlib.Path,
    succeed_on: int = 1,
    update_ok_on: int = 1,
):
    """Execute `script` with a mocked apt on PATH; return (rc, mock log)."""
    log.write_text("")
    for counter in ("n", "u"):
        (log.parent / counter).unlink(missing_ok=True)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "MOCK_LOG": str(log),
        "MOCK_N": str(log.parent / "n"),
        "MOCK_U": str(log.parent / "u"),
        "MOCK_SUCCEED_ON": str(succeed_on),
        "MOCK_UPDATE_OK_ON": str(update_ok_on),
        "DEBIAN_FRONTEND": "",
        # Always set on a real runner; some blocks append to them after the
        # install, and `set -u` would abort on an unset one.
        "GITHUB_WORKSPACE": str(ROOT),
        "GITHUB_PATH": str(log.parent / "github_path"),
        "GITHUB_ENV": str(log.parent / "github_env"),
        "GITHUB_OUTPUT": str(log.parent / "github_output"),
    }
    # The inline blocks run under `shell: bash`, whose default is -e -o pipefail.
    src = script if shell != "inline" else "set -euo pipefail\n" + script
    proc = subprocess.run(
        ["bash" if shell == "inline" else shell, "-c", src, "_", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    return proc.returncode, log.read_text()


def apt_steps() -> tuple[list[tuple[str, str]], list[str]]:
    """Every apt install step under .github/, split into (inline, via-helper).

    Inline steps are the ones invoking apt-get directly; each MUST carry the
    marker. Helper steps call apt-install.sh. Returning both lets the caller
    assert that marker discovery covers the whole tree.
    """
    inline: list[tuple[str, str]] = []
    helper: list[str] = []
    paths = sorted((ROOT / ".github/workflows").glob("*.yml"))
    paths += sorted((ROOT / ".github/actions").glob("*/action.yml"))
    for path in paths:
        doc = yaml.safe_load(path.read_text()) or {}
        jobs = doc.get("jobs") or {}
        # Composite actions keep their steps under runs.steps, not jobs.*.steps.
        if not jobs and isinstance(doc.get("runs"), dict):
            jobs = {path.parent.name: doc["runs"]}
        for job_name, job in jobs.items():
            for step in job.get("steps") or []:
                body = step.get("run") or ""
                where = (
                    f"{path.name if path.name != 'action.yml' else path.parent.name}:{job_name}"
                )
                if re.search(r"\bapt-get\s+install\b", body):
                    inline.append((where, body))
                elif "apt-install.sh" in body:
                    helper.append(where)
    return inline, helper


def main() -> int:
    helper_src = HELPER.read_text()
    inline, via_helper = apt_steps()

    with tempfile.TemporaryDirectory() as tmp:
        bindir = pathlib.Path(tmp) / "bin"
        bindir.mkdir()
        for name, body in STUBS.items():
            stub = bindir / name
            stub.write_text(body)
            stub.chmod(0o755)
        log = pathlib.Path(tmp) / "log"
        b = str(bindir)

        print(f"helper: {HELPER.relative_to(ROOT)}")
        for sh in ("bash", "sh", "dash"):
            if shutil.which(sh) is None:
                print(f"  SKIP  {sh} not installed")
                continue
            pkgs = ["bubblewrap", "tmux"]
            rc, out = run(helper_src, pkgs, sh, b, log, succeed_on=999)
            check(f"{sh}: persistent install failure -> 1", rc == 1, f"rc={rc}")
            check(
                f"{sh}: 3 attempts, 5s+10s backoff",
                out.count("apt-get install") == 3 and "sleep 5" in out and "sleep 10" in out,
                out,
            )
            rc, _ = run(helper_src, pkgs, sh, b, log, succeed_on=3)
            check(f"{sh}: recovers on 3rd install attempt -> 0", rc == 0, f"rc={rc}")

            # update failures must be retried exactly like install failures, and
            # must stop the install from running against a stale index.
            rc, out = run(helper_src, pkgs, sh, b, log, update_ok_on=999)
            check(f"{sh}: persistent update failure -> 1", rc == 1, f"rc={rc}")
            check(
                f"{sh}: update failure retried 3x, install never runs",
                out.count("apt-get update") == 3 and "apt-get install" not in out,
                out,
            )
            rc, out = run(helper_src, pkgs, sh, b, log, update_ok_on=3)
            check(
                f"{sh}: recovers on 3rd update attempt -> 0",
                rc == 0 and out.count("apt-get update") == 3 and out.count("apt-get install") == 1,
                f"rc={rc}\n{out}",
            )

            rc, _ = run(helper_src, [], sh, b, log)
            check(f"{sh}: no packages -> 2", rc == 2, f"rc={rc}")
            rc, _ = run(helper_src, ["--with-recommends"], sh, b, log)
            check(f"{sh}: --with-recommends with no packages -> 2", rc == 2, f"rc={rc}")
            rc, out = run(helper_src, ["bubblewrap"], sh, b, log)
            check(
                f"{sh}: default passes --no-install-recommends",
                rc == 0 and "--no-install-recommends bubblewrap" in out,
                out,
            )
            rc, out = run(helper_src, ["--with-recommends", "libmysqlclient-dev"], sh, b, log)
            check(
                f"{sh}: --with-recommends restores apt defaults",
                rc == 0 and "--no-install-recommends" not in out,
                out,
            )

        print(f"\ndiscovery: {len(inline)} inline, {len(via_helper)} via helper")
        check("some inline copies exist", bool(inline))
        check("some helper callers exist", bool(via_helper))
        unmarked = [w for w, body in inline if MARKER not in body]
        check(
            f"every direct apt-get install carries '{MARKER}'",
            not unmarked,
            f"unmarked: {unmarked}",
        )

        for where, body in inline:
            print(where)
            rc, out = run(body, [], "inline", b, log, succeed_on=999)
            check("persistent install failure -> 1", rc == 1, f"rc={rc}")
            check(
                "3 attempts, 5s+10s backoff",
                out.count("apt-get install") == 3 and "sleep 5" in out and "sleep 10" in out,
                out,
            )
            rc, out = run(body, [], "inline", b, log, update_ok_on=999)
            check("persistent update failure -> 1", rc == 1, f"rc={rc}")
            check(
                "update failure retried 3x, install never runs",
                out.count("apt-get update") == 3 and "apt-get install" not in out,
                out,
            )

            # Continuation: appending a sentinel proves the block BREAKS out of
            # the loop rather than exiting the step on success. Checking rc == 0
            # alone would pass even if `break` were replaced by `exit 0`.
            probe = f'{body}\necho "{SENTINEL}" >> "$MOCK_LOG"\n'
            rc, out = run(probe, [], "inline", b, log, succeed_on=3)
            check("recovers on 3rd attempt -> 0", rc == 0, f"rc={rc}\n{out}")
            check("execution continues past the loop", SENTINEL in out, out)
            # ...and that the step's own trailing commands really ran.
            tail = body.rsplit("\ndone", 1)[-1]
            expected = [
                c
                for c in STUBS
                if c not in ("sudo", "apt-get", "sleep")
                and re.search(rf"^\s*(sudo\s+)?{c}\b", tail, re.M)
            ]
            if expected:
                missing = [c for c in expected if f"{c} " not in out]
                check(
                    f"trailing commands ran ({', '.join(expected)})",
                    not missing,
                    f"missing: {missing}",
                )

            rc, out = run(body, [], "inline", b, log)
            has_recommends_flag = "--no-install-recommends" in out
            want_flag = "libmysqlclient-dev" not in body
            check(
                f"recommends flag {'present' if want_flag else 'absent'}",
                has_recommends_flag is want_flag,
                out,
            )

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + "; ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
