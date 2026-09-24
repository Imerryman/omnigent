"""Custom setuptools build for omnigent.

Generates ``omnigent/_build_info.py`` at wheel build time so the
CLI's update-check (``omnigent/update_check.py``) can tell the user
when their installed build is stale without having to consult
``git`` or hit a remote endpoint at startup.

All other build configuration lives in ``pyproject.toml``; this
file exists solely to register the cmdclass override that runs the
generator before ``build_py`` copies sources into the wheel.

The generated file is gitignored — it is recreated on every build
and only meaningful at install time, where it travels inside the
wheel alongside the rest of the package.

The same hook also builds the web SPA. Three environment variables
steer that build:

- ``OMNIGENT_SKIP_WEB_UI=true`` — CI opt-out; skip the web UI
  entirely (exact ``"true"`` only; set by our own workflows).
- ``OMNIGENT_BUILD_WEB_UI=1`` (also ``true``/``yes``/``force``/
  ``always``) — always rebuild, even when the on-disk bundle is
  already current.
- ``OMNIGENT_BUILD_WEB_UI=0`` (also ``false``/``no``/``never``/
  ``skip``) — never rebuild. For packagers and CI that ship their
  own prebuilt bundle and must not have the install shell out to
  pnpm.

Unset, the build is decided by a staleness check: the bundle dir
carries a stamp fingerprinting the frontend sources it was built
from, and a mismatch (or a missing stamp) triggers a rebuild.

``python setup.py --omnigent-build-web-ui`` runs that same build and
writes that same stamp from a checkout, without going through an
install. It exists because a bare ``pnpm --filter web run build``
produces a correct bundle with no stamp, which leaves the server's
staleness warning up and costs the next install a redundant rebuild.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

_MINIMUM_NODE_VERSION = (22, 13, 0)
_MINIMUM_NODE_VERSION_TEXT = ".".join(str(part) for part in _MINIMUM_NODE_VERSION)

# Build stamp written next to the Vite output. It lives *inside* the
# (gitignored) bundle dir so it ships in the wheel under the existing
# ``static/web-ui/**/*`` package-data glob, which lets the server read it
# back at startup. Keep the name in sync with
# ``omnigent/server/app.py``'s ``_WEB_UI_BUILD_STAMP_NAME``.
_WEB_UI_STAMP_NAME = "omnigent-build-stamp.json"
# Bumping this invalidates every existing stamp, forcing one rebuild.
_WEB_UI_STAMP_VERSION = 1
# World-readable: the installing user and the serving user are routinely
# different (a root/CI install serving as an unprivileged app user).
_WEB_UI_STAMP_MODE = 0o644
# Workspace-root files that change what the bundle contains. Named
# explicitly rather than discovered, because several are routinely
# gitignored yet load-bearing: ``.npmrc`` steers pnpm's resolution, and
# Vite reads ``.env*`` at build time and inlines the values into the
# bundle. Anything under ``web/`` is picked up by the walk instead.
_WEB_UI_ROOT_INPUTS = (
    ".env",
    ".env.local",
    ".env.production",
    ".env.production.local",
    ".npmrc",
    "package.json",
    "pnpm-lock.yaml",
    "pnpm-workspace.yaml",
)
# Directory names a tool owns outright, pruned at any depth. Reserved
# names only: npm owns ``node_modules``, git owns ``.git``, CPython owns
# ``__pycache__``, so a directory with one of these names is never
# hand-written source anywhere in the tree.
_WEB_UI_RESERVED_DIR_NAMES = frozenset({".git", "__pycache__", "node_modules"})
# Build output and tool caches, pruned by LOCATION relative to ``web/``,
# never by basename. A directory merely *named* ``dist`` or ``coverage``
# is not disposable: ``web/public/dist/banner.svg`` is a shipped asset,
# and pruning that basename everywhere would make it invisible to the
# fingerprint. Everything not listed here — all of ``web/public/**``
# included — is a build input. ``electron/build`` is absent on purpose:
# it is tracked source.
_WEB_UI_PRUNED_PATHS = frozenset(
    {
        ".turbo",
        ".vite",
        "android/.gradle",
        "android/build",
        "coverage",
        "dist",
        "dist-embed",
        "dist-ssr",
        "electron/dist",
        "ios/Pods",
        "ios/build",
        "storybook-static",
    }
)


class _GenerateBuildInfo(build_py):
    """Subclass of ``build_py`` that writes ``_build_info.py``.

    The override is the smallest possible intervention: run the
    generator, then defer to the stock ``build_py`` to copy sources
    (including the freshly-written ``_build_info.py``) into the
    wheel's build directory. No other behavior of the build is
    changed.
    """

    def run(self) -> None:
        """Build the web UI, generate ``_build_info.py``, then run build_py."""
        self._build_web_ui()
        self._write_build_info()
        super().run()
        self._bundle_examples()
        self._bundle_scripts()

    def _bundle_scripts(self) -> None:
        """Copy top-level maintenance scripts into package resources."""
        import shutil

        root = Path(__file__).resolve().parent
        src = root / "scripts" / "uninstall_oss.sh"
        if not src.is_file():
            return
        dest = Path(self.build_lib) / "omnigent" / "resources" / "scripts" / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    def _bundle_examples(self) -> None:
        """Copy bundled example agents into the wheel as real directories.

        ``omnigent/resources/examples/{polly,debby}`` may exist as symlinks
        into the top-level ``examples/`` tree (or not at all) depending on
        the checkout, and setuptools' ``package-data`` never materializes
        symlinks into the built wheel — a directory symlink is not walked.
        A plain ``pip install`` / ``uv tool install`` would then ship a
        package whose ``omnigent.resources.examples`` has no ``polly`` /
        ``debby`` subdir, and bare ``omnigent`` (first-run default → polly)
        dies with "Agent path not found".

        Fix: after ``build_py`` has populated ``build_lib``, copy the real
        example trees from the top-level ``examples/`` dir (present in every
        checkout) into
        ``build_lib/omnigent/resources/examples/<name>`` so every wheel is
        self-contained. This honors the contract documented in cli.py's
        ``_bundled_polly_path``: a symlink in a checkout, a real directory in
        an installed wheel. Editable installs (``uv sync``) resolve the
        in-checkout symlink directly and don't need this.
        """
        import shutil

        root = Path(__file__).resolve().parent
        dest_root = Path(self.build_lib) / "omnigent" / "resources" / "examples"
        for name in ("debby", "polly"):
            src = root / "examples" / name
            if not src.is_dir():
                continue
            dst = dest_root / name
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            elif dst.is_dir():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)

    def _build_web_ui(self) -> None:
        """Build and stamp the web SPA for this checkout.

        Thin wrapper so the policy lives in one module-level function
        that the manual entry point can call too; see
        :func:`_build_and_stamp_web_ui` for the whole contract.

        :raises SystemExit: As :func:`_build_and_stamp_web_ui` does.
        """
        _build_and_stamp_web_ui(_project_root())

    def _write_build_info(self) -> None:
        """Write ``omnigent/_build_info.py`` into the source tree.

        Writing to the source tree (rather than directly into the
        build dir) means editable installs (``pip install -e .``,
        ``uv sync``) also get the file — they're a single
        ``build_py`` invocation against an in-place package — and
        any later non-build code path that does ``from omnigent
        import _build_info`` works without re-running the build.
        """
        # Keep generated names and types aligned with omnigent/_build_info.pyi.
        target = Path(__file__).resolve().parent / "omnigent" / "_build_info.py"
        commit = _git_sha()
        # Use repr() for the SHA so quoting is always correct, even
        # for an empty fallback. The format is deliberately minimal
        # — anything more elaborate (version strings, branch names)
        # belongs in pyproject.toml or git tags, not here.
        target.write_text(
            '"""Auto-generated at wheel build time; do not edit.\n\n'
            "This module is created by ``setup.py`` immediately before\n"
            "``build_py`` packages the wheel, and is gitignored so it\n"
            "is recreated on every build. Consumers should import it\n"
            "defensively (``try: from omnigent import _build_info``)\n"
            "because source checkouts that have never been built will\n"
            "not have it on disk.\n"
            '"""\n'
            "from __future__ import annotations\n\n"
            f"BUILD_TIME_EPOCH: int = {int(time.time())}\n"
            f"COMMIT_SHA: str = {commit!r}\n"
        )


def _build_and_stamp_web_ui(
    root: Path,
    *,
    mode: str | None = None,
    respect_ci_skip: bool = True,
) -> None:
    """Build the web SPA into ``omnigent/server/static/web-ui/``.

    The server mounts that directory at ``/`` when present
    (``omnigent/server/app.py``); when absent it serves an
    API-only JSON landing page and the web UI is unreachable.
    The bundle is Vite build output, not tracked in git, so a
    plain ``pip install .`` / ``uv tool install`` from a checkout
    would otherwise ship no UI — the single most common "the web
    UI doesn't load" report.

    ``web/`` is a package in a pnpm workspace (``pnpm-workspace.yaml``
    and ``pnpm-lock.yaml`` at the repo root, ``packageManager:
    pnpm@11.15.1`` in the root ``package.json``), so the install and
    build run against the **workspace root** with ``--filter web``,
    matching ``deploy/databricks/build.sh`` and the CI workflows.
    Running ``pnpm install`` from inside ``web/`` would miss the
    committed lockfile and resolve against ``package.json`` alone —
    the legacy npm path that hit peer-dependency conflicts.

    Build policy, chosen to fix that case without slowing the
    backend-only dev loop, breaking node-less CI, or ever serving
    a bundle older than the frontend sources it came from:

    - Skip if ``web/`` is absent (sdists that don't vendor it).
    - Skip if ``OMNIGENT_SKIP_WEB_UI=true``. The hardened CI
      runners ship pnpm but have no fast registry mirror
      configured for the lint/test shards, so ``pnpm install``
      crawls against the public registry and hits the 600s
      timeout — 10 wasted minutes per ``uv sync`` for a bundle
      those jobs never serve. They set this env var to opt out.
    - Skip if ``OMNIGENT_BUILD_WEB_UI`` is ``0``/``false``/``no``/
      ``never``/``skip``. For packagers and downstream CI that
      ship a prebuilt bundle and must not shell out to pnpm.
    - Rebuild unconditionally if ``OMNIGENT_BUILD_WEB_UI`` is
      ``1``/``true``/``yes``/``force``/``always``.
    - Otherwise decide by staleness: hash the frontend build
      inputs (see :func:`_web_ui_source_files`) and compare
      against the stamp the last build left in the bundle dir.
      Equal means the bundle is current and the multi-minute Vite
      build is skipped; absent or different means rebuild. Content
      hashes, not mtimes: a git checkout rewrites mtimes freely, so
      mtimes cannot distinguish "same sources" from "different
      sources".

      The failure this replaces: the old rule was "skip if
      ``index.html`` exists", so an upgrade regenerated
      ``_build_info.py`` with the new commit while leaving a
      months-old SPA in place. The version string then lied about
      the frontend, and features merged since that bundle was
      built were simply missing from the browser.
    - When a rebuild is needed but the toolchain cannot run it,
      the outcome turns on one question only — is there already a
      bundle to fall back on?

      - No bundle — hard failure. Omnigent needs Node 22 LTS +
        pnpm at runtime anyway (the Claude / Codex / Pi harness
        CLIs are npm packages, and the web UI is a pnpm
        workspace), so a node-less machine gets a broken install
        either way; failing here, with a message that says how to
        fix it, beats a silent API-only install that surfaces
        later as "the web UI doesn't load".
      - A bundle exists — a loud stderr warning, the existing
        bundle is preserved, and the install continues. This holds
        under ``OMNIGENT_BUILD_WEB_UI=1`` too: forcing bypasses
        the *freshness check*, not the laws of physics, and
        aborting an upgrade over a frontend the user may not even
        use is worse than serving the old one provided we say so
        unmissably. The stamp is deliberately left untouched so
        the server repeats the warning at every startup.

    A failing ``pnpm install`` / ``pnpm --filter web run build``
    always aborts the install, bundle or no bundle: that is a real
    error in the sources, not a missing toolchain. No stamp is
    written on that path, so a half-written bundle is never
    certified as current.

    The pnpm commands run with ``COREPACK_ENABLE_DOWNLOAD_PROMPT=0``
    and no stdin: corepack (whether reached via its ``pnpm`` shim or
    as ``corepack pnpm``) otherwise blocks on an interactive
    confirmation before downloading the pinned pnpm, which a
    non-interactive install can never answer.

    :param root: Workspace root (the directory holding ``setup.py``).
    :param mode: Overrides the ``OMNIGENT_BUILD_WEB_UI`` decision;
        ``None`` reads it from the environment.
    :param respect_ci_skip: When false, ``OMNIGENT_SKIP_WEB_UI`` is
        ignored. The manual entry point passes false: someone who typed
        the build command meant it, whatever the shell exports.
    :raises SystemExit: If the web UI build fails, or if the
        toolchain (Node.js 22.13+, pnpm) is unavailable while a
        rebuild is required and no usable bundle is on disk to fall
        back on.
    """
    web_src = root / "web"
    bundle_dir = root / "omnigent" / "server" / "static" / "web-ui"
    bundle = bundle_dir / "index.html"
    stamp_path = bundle_dir / _WEB_UI_STAMP_NAME

    if not (web_src / "package.json").is_file():
        return
    # CI opt-out: exact "true" only — this is set by our own
    # workflows, not user-facing config.
    if respect_ci_skip and os.environ.get("OMNIGENT_SKIP_WEB_UI") == "true":
        return

    if mode is None:
        mode = _web_ui_build_mode(os.environ.get("OMNIGENT_BUILD_WEB_UI"))
    bundle_exists = bundle.is_file()
    if mode == "never" and not bundle_exists:
        return

    fingerprint = _web_ui_source_fingerprint(root)
    stamp = _read_web_ui_stamp(stamp_path) if bundle_exists else None
    current = bundle_exists and stamp is not None and stamp.get("sources") == fingerprint

    if mode == "never" or (mode == "auto" and current):
        # Nothing to build. Refresh the stamp when the bundle really is
        # current so its recorded commit tracks HEAD and the server's
        # startup check doesn't cry stale over an unchanged frontend.
        if current:
            _write_web_ui_stamp(stamp_path, sources=fingerprint, commit=_git_sha())
        return

    # pnpm 11.15.1 requires Node 22.13 and is stricter than Vite 8 and
    # oxlint. Enforce the effective toolchain floor before invoking pnpm.
    # Any existing bundle downgrades a missing toolchain from a failed
    # install to a loud warning; see the policy notes above.
    try:
        node_version = _require_supported_node()
        pnpm_cmd = _resolve_pnpm_command()
    except SystemExit as exc:
        # Forcing bypasses the freshness check, not a missing toolchain: an
        # existing bundle is preserved rather than aborting the install.
        if not bundle_exists:
            raise
        _warn_stale_web_ui_bundle(str(exc))
        return
    # A ``pnpm`` on PATH is often a corepack shim, which asks "Do you
    # want to continue? [Y/n]" before fetching the pinned pnpm. That
    # prompt is invisible under a build backend, so the install looks
    # hung; "0" downloads without asking (shims default it to "1").
    env = {**os.environ, "COREPACK_ENABLE_DOWNLOAD_PROMPT": "0"}
    # Invalidate BEFORE anything can touch live bundle output. Vite writes
    # straight into the bundle dir, so a build that damages the SPA and then
    # fails must not leave a stamp behind — with unchanged sources the next
    # automatic install would match that stamp and skip, serving the damaged
    # bundle forever. No stamp always means rebuild, which is the safe
    # direction, and the server warns in the meantime.
    stamp_path.unlink(missing_ok=True)
    try:
        # Workspace root, not ``web/``: the lockfile and workspace
        # manifest live at the repo root. ``--frozen-lockfile``
        # matches CI and guarantees the build is reproducible from
        # the committed ``pnpm-lock.yaml``. No stdin, so nothing in
        # the toolchain can block on input we can never deliver.
        subprocess.run(
            [*pnpm_cmd, "install", "--frozen-lockfile", "--filter", "web"],
            cwd=root,
            check=True,
            timeout=600,
            stdin=subprocess.DEVNULL,
            env=env,
        )
        subprocess.run(
            [*pnpm_cmd, "--filter", "web", "run", "build"],
            cwd=root,
            check=True,
            timeout=600,
            stdin=subprocess.DEVNULL,
            env=env,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise SystemExit(
            f"omnigent build: web UI build failed on Node.js {node_version} "
            f"({_subprocess_failure_details(exc)}). Fix the failure above "
            "and rerun the install. If "
            "this Node.js release is incompatible with pnpm or Vite, "
            f"upgrade to Node.js {_MINIMUM_NODE_VERSION_TEXT} or newer and "
            "retry. To deliberately install "
            "without the web UI (API-only server), set "
            "OMNIGENT_SKIP_WEB_UI=true."
        ) from exc
    # Only now that the bundle on disk really came from these sources.
    _write_web_ui_stamp(stamp_path, sources=fingerprint, commit=_git_sha())


def _project_root() -> Path:
    """Return the directory holding this ``setup.py`` (repo or sdist root)."""
    return Path(__file__).resolve().parent


def _web_ui_build_mode(raw: str | None) -> str:
    """Map ``OMNIGENT_BUILD_WEB_UI`` onto a build decision.

    :param raw: Raw environment value, or ``None`` when unset.
    :returns: ``"force"`` (always rebuild), ``"never"`` (never rebuild),
        or ``"auto"`` (rebuild only when the bundle is stale). Anything
        unrecognized falls through to ``"auto"`` rather than failing the
        install over a typo.
    """
    value = (raw or "").strip().lower()
    if value in ("1", "true", "yes", "force", "always"):
        return "force"
    if value in ("0", "false", "no", "never", "skip"):
        return "never"
    return "auto"


def _resolve_pnpm_command() -> list[str]:
    """Return the argv prefix that runs pnpm, or abort with instructions.

    pnpm first; fall back to corepack (bundled with Node 22+), which
    downloads the pnpm version pinned by the root ``package.json``'s
    ``packageManager`` field on first use.

    :returns: The command prefix, e.g. ``["pnpm"]`` or ``["corepack", "pnpm"]``.
    :raises SystemExit: If neither pnpm nor corepack is on PATH.
    """
    import shutil

    pnpm = shutil.which("pnpm")
    if pnpm is not None:
        return [pnpm]
    corepack = shutil.which("corepack")
    if corepack is not None:
        return [corepack, "pnpm"]
    raise SystemExit(
        "omnigent build: pnpm not found on PATH, so the web UI "
        "cannot be built. Omnigent requires Node.js 22 LTS or "
        "newer with pnpm (the web UI is a pnpm workspace; the "
        "Claude / Codex / Pi harness CLIs are npm packages). "
        "Install Node from https://nodejs.org/en/download and "
        "enable pnpm with `corepack enable` (or `npm install -g "
        "pnpm`), then rerun the install. To deliberately install "
        "without the web UI (API-only server), set "
        "OMNIGENT_SKIP_WEB_UI=true."
    )


def _warn_stale_web_ui_bundle(reason: str) -> None:
    """Shout on stderr that the install is leaving a stale SPA in place.

    This is the last chance to tell the user before the server starts
    serving a frontend older than the version it reports, so it is
    deliberately hard to miss in a wall of pip/uv output.

    :param reason: Why the rebuild could not run, quoted into the banner.
    """
    rule = "!" * 78
    sys.stderr.write(
        f"\n{rule}\n"
        "!! omnigent build: THE WEB UI BUNDLE IS STALE AND WAS NOT REBUILT.\n"
        f"!!\n!! Why: {reason}\n!!\n"
        "!! The server will serve an OUT-OF-DATE web UI while reporting the\n"
        "!! newly installed version. Anything merged since that bundle was\n"
        "!! built will simply be missing from the browser.\n"
        "!!\n"
        "!! Fix it with either of:\n"
        "!!   python setup.py --omnigent-build-web-ui     (from a checkout)\n"
        "!!   OMNIGENT_BUILD_WEB_UI=1 <your install command>\n"
        "!!\n"
        "!! A bare `pnpm --filter web run build` fixes the bundle but writes\n"
        "!! no build stamp, so the server keeps warning and the next install\n"
        "!! rebuilds again. Both commands above write the stamp.\n"
        "!!\n"
        "!! To ship your own prebuilt bundle and silence this deliberately,\n"
        "!! set OMNIGENT_BUILD_WEB_UI=0.\n"
        f"{rule}\n\n"
    )
    sys.stderr.flush()


def _web_ui_build_inputs(root: Path) -> tuple[list[Path], dict[str, str]]:
    """Return everything whose content or shape determines the bundle.

    Build inputs are defined here positively, and explicitly NOT by
    git's ignore rules. ``.gitignore`` answers "should this be
    committed?", which is a different question from "does this change
    the bundle?", and the two come apart in ways that would reintroduce
    the false SKIP this whole mechanism exists to prevent:

    - A gitignored file can be load-bearing. Vite reads ``web/.env*``
      at build time and inlines the values; a generated-but-ignored
      module is still imported by the sources that import it.
    - ``git ls-files`` reports a directory symlink as a single entry,
      not as the tree behind it, so edits inside an imported symlinked
      source dir would be invisible.

    So the primary enumeration is :func:`_walk_web_ui_sources`, an
    ignore-agnostic walk that prunes by location rather than basename and
    records symlink topology. ``git ls-files`` is unioned on top purely
    as a safety net; it is never the sole source of truth, and its
    absence changes nothing that matters. Both enumerations are held to
    the same pruning policy, so what counts as build output is decided in
    one place — otherwise a ``node_modules`` that simply was not
    gitignored would slip back in through the union.

    :param root: Workspace root (the directory holding ``setup.py``).
    :returns: ``(files, links)`` — deduplicated absolute paths, unordered
        (the caller sorts), and the symlink topology behind them.
    """
    candidates: dict[Path, None] = {root / name: None for name in _WEB_UI_ROOT_INPUTS}
    walked, links = _walk_web_ui_sources(root / "web")
    for path in walked:
        candidates[path] = None
    for path in _git_listed_files(root, "web") or ():
        if not _is_pruned_web_ui_path(root, path):
            candidates[path] = None
    return [path for path in candidates if path.is_file()], links


def _is_pruned_web_ui_path(root: Path, path: Path) -> bool:
    """Whether ``path`` sits in a location the walk would have pruned.

    Applied to the git listing so both enumerations answer "is this a
    build input?" from the single policy in
    :data:`_WEB_UI_RESERVED_DIR_NAMES` / :data:`_WEB_UI_PRUNED_PATHS`.

    :param root: Workspace root (the directory holding ``setup.py``).
    :param path: Absolute path to classify.
    :returns: True when any directory component is pruned.
    """
    try:
        directories = path.relative_to(root / "web").parts[:-1]
    except ValueError:
        return False
    if any(name in _WEB_UI_RESERVED_DIR_NAMES for name in directories):
        return True
    return any(
        "/".join(directories[: depth + 1]) in _WEB_UI_PRUNED_PATHS
        for depth in range(len(directories))
    )


def _git_listed_files(root: Path, subdir: str) -> list[Path] | None:
    """Return git's view of ``subdir``'s files, or ``None``.

    A safety net for the walk, not a definition of build inputs — see
    :func:`_web_ui_build_inputs`.

    :param root: Repository root to run git in.
    :param subdir: Pathspec to limit the listing to.
    :returns: Tracked plus not-ignored-untracked files, or ``None`` when
        git is absent, fails, or this is not a work tree.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                subdir,
            ],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return [root / os.fsdecode(entry) for entry in result.stdout.split(b"\0") if entry]


def _walk_web_ui_sources(web: Path) -> tuple[list[Path], dict[str, str]]:
    """Walk ``web/`` for build inputs, pruning output by location.

    Ignore-agnostic by design: dotfiles and gitignored files are
    included, because Vite happily reads both. Pruning is by reserved
    basename (:data:`_WEB_UI_RESERVED_DIR_NAMES`) or by exact location
    (:data:`_WEB_UI_PRUNED_PATHS`) — never by an arbitrary basename at
    arbitrary depth, which would swallow real inputs.

    Directory symlinks are followed, since an imported source tree may
    be one, and traversal is guarded by the real directories already
    visited so a link pointing at an ancestor cannot loop. That guard
    also skips a *sibling alias* — two links onto one target — which on
    its own would hide a retarget: flipping ``src/selected`` from
    ``src/a`` to ``src/b`` visits neither, because both were already
    traversed under their own names. So every symlink's target is
    recorded as topology alongside the file list, and a retarget changes
    the digest even though no file content did. Recording the edge is
    much cheaper than re-traversing each alias, and cannot blow up on a
    legitimately shared directory.

    Paths are reported as traversed rather than as resolved, so a
    symlink into a tree outside the workspace still yields a
    workspace-relative name and the fingerprint stays machine
    independent.

    :param web: The ``web/`` package directory.
    :returns: ``(files, links)`` — absolute paths of candidate build
        inputs, and a mapping of each symlink's traversal path to its raw
        target as ``os.readlink`` reports it.
    """
    found: list[Path] = []
    links: dict[str, str] = {}
    visited: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(web, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in visited:
            dirnames[:] = []
            continue
        visited.add(real)
        relative = os.path.relpath(dirpath, web)
        prefix = "" if relative == os.curdir else relative.replace(os.sep, "/") + "/"
        # Sorted so which of two links to one target is traversed first
        # (and therefore under which name its files are hashed) is stable.
        kept = []
        for name in sorted(dirnames):
            if name in _WEB_UI_RESERVED_DIR_NAMES:
                continue
            if f"{prefix}{name}" in _WEB_UI_PRUNED_PATHS:
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in sorted([*kept, *filenames]):
            entry = Path(dirpath) / name
            if entry.is_symlink():
                try:
                    links[f"{prefix}{name}"] = os.readlink(entry)
                except OSError:
                    links[f"{prefix}{name}"] = "<unreadable>"
        found.extend(Path(dirpath) / name for name in filenames)
    return found, links


def _web_ui_source_fingerprint(root: Path) -> str:
    """Return a content digest of the frontend inputs.

    Content hashes only — mtimes are useless here because a git checkout
    or a fresh clone rewrites them without changing a byte of source.

    :param root: Workspace root (the directory holding ``setup.py``).
    :returns: Hex sha256 over ``(relative path, file digest)`` pairs plus
        the symlink topology, so retargeting a link changes the digest
        even when no file content did.
    """
    files, links = _web_ui_build_inputs(root)
    digest = hashlib.sha256()
    digest.update(f"stamp-v{_WEB_UI_STAMP_VERSION}\n".encode())
    for path in sorted(files):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        try:
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        except OSError:
            # Raced away or unreadable; record its absence rather than
            # silently hashing the same value as an empty file.
            digest.update(b"<unreadable>")
        digest.update(b"\0")
    digest.update(b"symlinks\0")
    for name in sorted(links):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(links[name].encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_web_ui_stamp(path: Path) -> dict[str, object] | None:
    """Return the parsed build stamp, or ``None`` when unusable.

    A missing, corrupt, or older-format stamp all mean the same thing to
    the caller: we cannot prove the bundle is current, so rebuild.

    :param path: Path to the stamp file inside the bundle dir.
    :returns: The stamp mapping, or ``None``.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("stamp_version") != _WEB_UI_STAMP_VERSION:
        return None
    return data


def _write_web_ui_stamp(path: Path, *, sources: str, commit: str) -> None:
    """Record what the bundle on disk was built from.

    ``sources`` drives the build-time staleness check; ``commit`` is what
    the server compares against ``_build_info.COMMIT_SHA`` at startup.

    :param path: Path to the stamp file inside the bundle dir.
    :param sources: Frontend-source fingerprint from
        :func:`_web_ui_source_fingerprint`.
    :param commit: Git HEAD SHA, or ``""`` when git is unavailable.
    """
    payload = {
        "built_at": int(time.time()),
        "commit": commit,
        "sources": sources,
        "stamp_version": _WEB_UI_STAMP_VERSION,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish atomically so a concurrent reader (the server's startup check)
    # can never observe a truncated stamp. Known limitation: this does not
    # serialize two *builds* racing on the same tree — the last writer wins,
    # and the loser's bundle may end up described by the winner's stamp.
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    # NamedTemporaryFile creates 0600 and os.replace preserves the mode, so
    # without this the stamp is unreadable to a server running as a different
    # user than the installer — which reads back as a permanent "no readable
    # build stamp" warning on a perfectly fresh bundle.
    os.chmod(temporary, _WEB_UI_STAMP_MODE)
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _require_supported_node() -> str:
    """Return the Node.js version or abort when it is older than 22.13.

    The pinned pnpm 11.15.1 requires Node 22.13. Newer Node releases
    remain valid.

    :returns: The normalized Node.js version string.
    :raises SystemExit: If ``node`` is missing, ``node --version`` fails,
        or the reported version is older than 22.13.
    """
    import shutil

    node = shutil.which("node")
    if node is None:
        raise SystemExit(
            "omnigent build: Node.js not found on PATH, so the web UI "
            f"cannot be built. Node.js {_MINIMUM_NODE_VERSION_TEXT} or newer "
            "is required. Install it from https://nodejs.org/en/download and "
            "retry, or run with OMNIGENT_SKIP_WEB_UI=true to skip the web UI "
            "build."
        )
    try:
        result = subprocess.run(
            [node, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise SystemExit(
            f"omnigent build: could not determine the Node.js version "
            f"(`node --version` failed: {exc}). Node.js "
            f"{_MINIMUM_NODE_VERSION_TEXT} or newer is required. Install it "
            "from https://nodejs.org/en/download and retry, or run with "
            "OMNIGENT_SKIP_WEB_UI=true to skip the web UI build."
        ) from exc
    # ``node --version`` prints ``v22.14.0``.
    version_str = result.stdout.strip().lstrip("v")
    try:
        version_parts = tuple(int(part) for part in version_str.split(".")[:3])
        if len(version_parts) != 3:
            raise ValueError
    except ValueError:
        raise SystemExit(
            f"omnigent build: could not parse Node.js version "
            f"{version_str or 'unknown'}. Node.js "
            f"{_MINIMUM_NODE_VERSION_TEXT} or newer is required. Install it "
            "from https://nodejs.org/en/download and retry, or run with "
            "OMNIGENT_SKIP_WEB_UI=true to skip the web UI build."
        ) from None
    if version_parts < _MINIMUM_NODE_VERSION:
        raise SystemExit(
            f"omnigent build: Node.js {_MINIMUM_NODE_VERSION_TEXT} or newer "
            f"is required but found Node.js {version_str}. Upgrade from "
            "https://nodejs.org/en/download and retry, or run with "
            "OMNIGENT_SKIP_WEB_UI=true to skip the web UI build."
        )
    return version_str


def _subprocess_failure_details(exc: BaseException) -> str:
    """Return an actionable subprocess failure summary, including stderr."""
    details = str(exc)
    stderr = getattr(exc, "stderr", None)
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    if isinstance(stderr, str):
        stderr = stderr.strip()
        if stderr and stderr not in details:
            details = f"{details}; stderr: {stderr}"
    return details


def _git_sha() -> str:
    """Return the current Git HEAD SHA, or empty string on failure.

    Empty-string fallback is intentional: when this is run inside a
    Docker build context with no ``git`` binary, or when the build
    happens from an sdist that has no ``.git/`` directory, the field
    must still be populated with a stable string so the generated
    module remains importable. The CLI update-check treats an empty
    SHA as "no commit info available" and silently falls back to
    timestamp-only nag logic.

    :returns: 40-character full hex SHA, or ``""`` on any failure.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return ""
    return result.stdout.strip()


_MANUAL_BUILD_FLAG = "--omnigent-build-web-ui"


def _manual_build_requested(argv: list[str]) -> bool:
    """Whether ``argv`` asks for the manual build, not a setuptools command.

    :param argv: Full process argv, ``argv[0]`` being the script.
    :returns: True only for the sentinel flag, so every ordinary
        setuptools invocation (``--help``, ``egg_info``, ``bdist_wheel``)
        falls through to ``setup()`` untouched.
    """
    return _MANUAL_BUILD_FLAG in argv[1:]


if _manual_build_requested(sys.argv):
    # Manual entry point: run exactly the build an install runs and write
    # exactly the same stamp, so the server's staleness warning clears and
    # the next install takes the fast path. A bare `pnpm --filter web run
    # build` does neither. Handled before setup() so setuptools never sees
    # an argument it would reject.
    _build_and_stamp_web_ui(_project_root(), mode="force", respect_ci_skip=False)
    raise SystemExit(0)

setup(cmdclass={"build_py": _GenerateBuildInfo})
