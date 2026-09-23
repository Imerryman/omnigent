"""Tests for the web UI build prerequisites in ``setup.py``."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest


def _load_setup_module() -> ModuleType:
    """Load repo-root ``setup.py`` without running the real ``setup()``.

    ``setup.py`` calls ``setup(...)`` at module scope, which would drive
    setuptools off pytest's argv; patch it to a no-op during load so we
    can exercise the module's helpers in isolation.
    """
    path = Path(__file__).resolve().parents[1] / "setup.py"
    spec = importlib.util.spec_from_file_location("_omnigent_setup_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    setuptools = ModuleType("setuptools")
    setuptools.setup = mock.Mock()  # type: ignore[attr-defined]
    setuptools_command = ModuleType("setuptools.command")
    setuptools_build_py = ModuleType("setuptools.command.build_py")
    setuptools_build_py.build_py = type("build_py", (), {})  # type: ignore[attr-defined]
    with mock.patch.dict(
        sys.modules,
        {
            "setuptools": setuptools,
            "setuptools.command": setuptools_command,
            "setuptools.command.build_py": setuptools_build_py,
        },
    ):
        spec.loader.exec_module(module)
    return module


def _run_node_version_check(module: ModuleType, version_output: str) -> str:
    """Invoke the Node.js gate with ``node --version`` stubbed."""
    completed = mock.Mock(stdout=version_output)
    with (
        mock.patch("shutil.which", return_value="/usr/bin/node"),
        mock.patch.object(module.subprocess, "run", return_value=completed),
    ):
        return module._require_supported_node()


def test_node_22_13_passes() -> None:
    module = _load_setup_module()
    assert _run_node_version_check(module, "v22.13.0\n") == "22.13.0"


@pytest.mark.parametrize("version", ["v23.0.0\n", "v24.14.0\n", "v25.2.1\n"])
def test_newer_node_versions_pass(version: str) -> None:
    module = _load_setup_module()
    assert _run_node_version_check(module, version) == version.strip().lstrip("v")


@pytest.mark.parametrize("version", ["v22.12.0\n", "v20.20.1\n"])
def test_older_node_fails(version: str) -> None:
    module = _load_setup_module()
    with pytest.raises(SystemExit) as excinfo:
        _run_node_version_check(module, version)
    message = str(excinfo.value)
    assert "Node.js 22.13.0 or newer is required" in message
    assert version.strip().lstrip("v") in message
    assert "https://nodejs.org/en/download" in message
    assert "OMNIGENT_SKIP_WEB_UI=true" in message


def test_unparseable_version_fails() -> None:
    module = _load_setup_module()
    with pytest.raises(SystemExit) as excinfo:
        _run_node_version_check(module, "not-a-version\n")
    message = str(excinfo.value)
    assert "could not parse Node.js version not-a-version" in message
    assert "Node.js 22.13.0 or newer is required" in message
    assert "https://nodejs.org/en/download" in message
    assert "OMNIGENT_SKIP_WEB_UI=true" in message


def test_node_missing_fails() -> None:
    module = _load_setup_module()
    with mock.patch("shutil.which", return_value=None):
        with pytest.raises(SystemExit) as excinfo:
            module._require_supported_node()
    message = str(excinfo.value)
    assert "Node.js not found on PATH" in message
    assert "Node.js 22.13.0 or newer is required" in message
    assert "https://nodejs.org/en/download" in message
    assert "OMNIGENT_SKIP_WEB_UI=true" in message


def test_node_version_probe_failure_fails() -> None:
    module = _load_setup_module()
    with (
        mock.patch("shutil.which", return_value="/usr/bin/node"),
        mock.patch.object(
            module.subprocess,
            "run",
            side_effect=OSError("boom"),
        ),
    ):
        with pytest.raises(SystemExit) as excinfo:
            module._require_supported_node()
    message = str(excinfo.value)
    assert "could not determine the Node.js version" in message
    assert "Node.js 22.13.0 or newer is required" in message
    assert "https://nodejs.org/en/download" in message
    assert "OMNIGENT_SKIP_WEB_UI=true" in message


def test_skip_web_ui_bypasses_node_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_setup_module()
    monkeypatch.setenv("OMNIGENT_SKIP_WEB_UI", "true")
    require_node = mock.Mock()
    with mock.patch.object(module, "_require_supported_node", require_node):
        module._GenerateBuildInfo._build_web_ui(object())
    require_node.assert_not_called()


def test_newer_node_build_failure_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_setup_module()
    monkeypatch.delenv("OMNIGENT_SKIP_WEB_UI", raising=False)
    monkeypatch.setenv("OMNIGENT_BUILD_WEB_UI", "1")
    failure = module.subprocess.CalledProcessError(
        1,
        ["pnpm", "install"],
        stderr="ERR_PNPM_UNSUPPORTED_ENGINE: use Node.js 22.13.0 or newer",
    )
    with (
        mock.patch.object(module, "_require_supported_node", return_value="25.2.1"),
        mock.patch("shutil.which", return_value="/usr/bin/pnpm"),
        mock.patch.object(module.subprocess, "run", side_effect=failure),
        pytest.raises(SystemExit) as excinfo,
    ):
        module._GenerateBuildInfo._build_web_ui(object())
    message = str(excinfo.value)
    assert "web UI build failed on Node.js 25.2.1" in message
    assert "ERR_PNPM_UNSUPPORTED_ENGINE" in message
    assert "upgrade to Node.js 22.13.0 or newer" in message
    assert "OMNIGENT_SKIP_WEB_UI=true" in message
    assert excinfo.value.__cause__ is failure


def test_real_node_floor_matches_pinned_pnpm() -> None:
    """Exercise the setup gate and pinned pnpm with real boundary binaries.

    Set both environment variables to opt in. Keeping this test explicit avoids
    downloading Node during ordinary pytest runs while providing a real
    toolchain check that cannot pass from mocked ``node --version`` output.
    """
    node_paths = {
        "22.12.0": os.environ.get("OMNIGENT_TEST_NODE_22_12"),
        "22.13.0": os.environ.get("OMNIGENT_TEST_NODE_22_13"),
    }
    if not any(node_paths.values()):
        pytest.skip(
            "set OMNIGENT_TEST_NODE_22_12 and OMNIGENT_TEST_NODE_22_13 "
            "to run the real toolchain boundary check"
        )
    assert all(node_paths.values()), "both real Node boundary binaries are required"

    root = Path(__file__).resolve().parents[1]
    package_manager = json.loads((root / "package.json").read_text())["packageManager"]
    assert package_manager == "pnpm@11.15.1"
    module = _load_setup_module()

    for expected_version, raw_node_path in node_paths.items():
        assert raw_node_path is not None
        node_path = Path(raw_node_path).resolve()
        assert node_path.is_file(), f"Node binary not found: {node_path}"
        version_result = subprocess.run(
            [node_path, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert version_result.stdout.strip() == f"v{expected_version}"

        with mock.patch("shutil.which", return_value=str(node_path)):
            if expected_version == "22.12.0":
                with pytest.raises(SystemExit):
                    module._require_supported_node()
            else:
                assert module._require_supported_node() == expected_version

        corepack_path = node_path.parent / "corepack"
        assert corepack_path.is_file(), f"Corepack not found: {corepack_path}"
        env = {
            **os.environ,
            "COREPACK_DEFAULT_TO_LATEST": "0",
            "COREPACK_ENABLE_DOWNLOAD_PROMPT": "0",
            "PATH": f"{node_path.parent}{os.pathsep}{os.environ['PATH']}",
        }
        pnpm_result = subprocess.run(
            [corepack_path, "pnpm", "--version"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        output = f"{pnpm_result.stdout}\n{pnpm_result.stderr}"
        if expected_version == "22.12.0":
            assert pnpm_result.returncode != 0
            assert "requires at least Node.js v22.13" in output
        else:
            assert pnpm_result.returncode == 0, output
            assert "11.15.1" in pnpm_result.stdout


# --- Web UI bundle staleness -------------------------------------------------
#
# ``setup.py`` must never leave an install serving a bundle older than the
# frontend sources it was built from, while still skipping the multi-minute
# Vite build when nothing about the frontend changed. These exercise that
# decision table with the real pnpm/vite invocation stubbed out.

_FAKE_SHA = "a" * 40
_BUILD_ENV_VARS = ("OMNIGENT_SKIP_WEB_UI", "OMNIGENT_BUILD_WEB_UI")


def _make_project(tmp_path: Path, *, bundle: bool = False) -> Path:
    """Create a minimal workspace layout that ``_build_web_ui`` understands."""
    root = tmp_path / "proj"
    (root / "web" / "src").mkdir(parents=True)
    (root / "web" / "package.json").write_text('{"name": "web"}\n')
    (root / "web" / "src" / "main.tsx").write_text("export const app = 1;\n")
    (root / "package.json").write_text('{"packageManager": "pnpm@11.15.1"}\n')
    (root / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n")
    if bundle:
        bundle_dir = root / "omnigent" / "server" / "static" / "web-ui"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "index.html").write_text("<!doctype html>\n")
    return root


def _git_init(root: Path, *, ignore: str | None = None) -> None:
    """Make ``root`` a real work tree with everything present committed."""
    if ignore is not None:
        (root / ".gitignore").write_text(ignore)
    commands: tuple[list[str], ...] = (
        ["init", "-q"],
        ["config", "user.email", "test@example.com"],
        ["config", "user.name", "Test"],
        ["add", "-A"],
        ["commit", "-q", "-m", "initial"],
    )
    for args in commands:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _stamp_path(root: Path) -> Path:
    """Path of the build stamp inside ``root``'s bundle dir."""
    return root / "omnigent" / "server" / "static" / "web-ui" / "omnigent-build-stamp.json"


def _run_build(
    module: ModuleType,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    node: bool = True,
    pnpm: bool = True,
    sha: str = _FAKE_SHA,
    env: dict[str, str] | None = None,
    real_git: bool = False,
) -> list[list[str]]:
    """Invoke ``_build_web_ui`` against ``root``, returning the pnpm argvs.

    The real ``pnpm install`` / ``pnpm --filter web run build`` never runs;
    the point is which commands the policy decides to issue.
    """
    for name in _BUILD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(module, "_project_root", lambda: root)
    monkeypatch.setattr(module, "_git_sha", lambda: sha)
    if not real_git:
        # Force the walk-only path so the outcome cannot depend on whether the
        # tmp dir happens to sit inside a work tree. ``real_git=True`` leaves
        # the git union live and drives the decision through it for real.
        monkeypatch.setattr(module, "_git_listed_files", lambda *_a, **_k: None)

    def _node() -> str:
        if not node:
            raise SystemExit("omnigent build: Node.js not found on PATH, ...")
        return "22.20.0"

    def _pnpm() -> list[str]:
        if not pnpm:
            raise SystemExit("omnigent build: pnpm not found on PATH, ...")
        return ["pnpm"]

    monkeypatch.setattr(module, "_require_supported_node", _node)
    monkeypatch.setattr(module, "_resolve_pnpm_command", _pnpm)

    invocations: list[list[str]] = []
    real_run = module.subprocess.run

    def _fake_run(cmd: list[str], **kwargs: object) -> object:
        # git still runs for real: it is part of the code under test when
        # ``real_git`` is set, and never a build command worth recording.
        if cmd and str(cmd[0]) == "git":
            return real_run(cmd, **kwargs)
        invocations.append([str(part) for part in cmd])
        return mock.Mock(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    module._GenerateBuildInfo._build_web_ui(object())
    return invocations


def test_missing_bundle_builds_and_stamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_setup_module()
    root = _make_project(tmp_path)
    invocations = _run_build(module, root, monkeypatch)
    assert invocations == [
        ["pnpm", "install", "--frozen-lockfile", "--filter", "web"],
        ["pnpm", "--filter", "web", "run", "build"],
    ]
    stamp = json.loads(_stamp_path(root).read_text())
    assert stamp["commit"] == _FAKE_SHA
    assert stamp["sources"] == module._web_ui_source_fingerprint(root)


def test_matching_stamp_skips_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    assert _run_build(module, root, monkeypatch)  # first pass builds + stamps
    assert _run_build(module, root, monkeypatch) == []


def test_missing_stamp_rebuilds_existing_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression this replaces: ``index.html`` alone proves nothing."""
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    assert _run_build(module, root, monkeypatch)


def test_changed_frontend_source_rebuilds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    _run_build(module, root, monkeypatch)
    (root / "web" / "src" / "main.tsx").write_text("export const app = 2;\n")
    assert _run_build(module, root, monkeypatch)


def test_changed_lockfile_rebuilds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    _run_build(module, root, monkeypatch)
    (root / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n# bumped\n")
    assert _run_build(module, root, monkeypatch)


def test_new_frontend_source_file_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    _run_build(module, root, monkeypatch)
    (root / "web" / "src" / "ModelPicker.tsx").write_text("export const p = 1;\n")
    assert _run_build(module, root, monkeypatch)


def test_corrupt_stamp_rebuilds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    _run_build(module, root, monkeypatch)
    _stamp_path(root).write_text("{not json")
    assert _run_build(module, root, monkeypatch)


def test_stamp_version_bump_invalidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    _run_build(module, root, monkeypatch)
    stamp = json.loads(_stamp_path(root).read_text())
    stamp["stamp_version"] = module._WEB_UI_STAMP_VERSION + 1
    _stamp_path(root).write_text(json.dumps(stamp))
    assert _run_build(module, root, monkeypatch)


def test_current_bundle_refreshes_stamp_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unchanged frontend on a new commit must not look stale at runtime."""
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    _run_build(module, root, monkeypatch, sha="b" * 40)
    assert _run_build(module, root, monkeypatch, sha="c" * 40) == []
    assert json.loads(_stamp_path(root).read_text())["commit"] == "c" * 40


@pytest.mark.parametrize("value", ["1", "true", "yes", "force", "ALWAYS"])
def test_force_rebuilds_a_current_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    _run_build(module, root, monkeypatch)
    assert _run_build(module, root, monkeypatch, env={"OMNIGENT_BUILD_WEB_UI": value})


@pytest.mark.parametrize("value", ["0", "false", "no", "never", "SKIP"])
def test_opt_out_leaves_a_stale_bundle_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Packagers shipping their own bundle must not trigger a pnpm build."""
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    invocations = _run_build(module, root, monkeypatch, env={"OMNIGENT_BUILD_WEB_UI": value})
    assert invocations == []
    # No stamp is invented for a bundle whose provenance we cannot vouch for.
    assert not _stamp_path(root).is_file()


def test_opt_out_without_a_bundle_builds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_setup_module()
    root = _make_project(tmp_path)
    invocations = _run_build(module, root, monkeypatch, env={"OMNIGENT_BUILD_WEB_UI": "never"})
    assert invocations == []


def test_opt_out_still_refreshes_a_current_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    _run_build(module, root, monkeypatch, sha="b" * 40)
    invocations = _run_build(
        module,
        root,
        monkeypatch,
        sha="c" * 40,
        env={"OMNIGENT_BUILD_WEB_UI": "0"},
    )
    assert invocations == []
    assert json.loads(_stamp_path(root).read_text())["commit"] == "c" * 40


def test_skip_web_ui_wins_over_staleness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    invocations = _run_build(module, root, monkeypatch, env={"OMNIGENT_SKIP_WEB_UI": "true"})
    assert invocations == []


@pytest.mark.parametrize("missing", ["node", "pnpm"])
def test_stale_bundle_without_toolchain_warns_instead_of_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    missing: str,
) -> None:
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    invocations = _run_build(
        module,
        root,
        monkeypatch,
        node=missing != "node",
        pnpm=missing != "pnpm",
    )
    assert invocations == []
    stderr = capsys.readouterr().err
    assert "THE WEB UI BUNDLE IS STALE AND WAS NOT REBUILT" in stderr
    assert "pnpm --filter web run build" in stderr
    assert "OMNIGENT_BUILD_WEB_UI=1" in stderr
    assert "OMNIGENT_BUILD_WEB_UI=0" in stderr
    assert f"{'Node.js' if missing == 'node' else 'pnpm'} not found on PATH" in stderr
    # Leaving the stamp absent keeps the server warning at every startup.
    assert not _stamp_path(root).is_file()


@pytest.mark.parametrize("missing", ["node", "pnpm"])
def test_missing_bundle_without_toolchain_still_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    module = _load_setup_module()
    root = _make_project(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        _run_build(
            module,
            root,
            monkeypatch,
            node=missing != "node",
            pnpm=missing != "pnpm",
        )
    assert "not found on PATH" in str(excinfo.value)


@pytest.mark.parametrize("missing", ["node", "pnpm"])
def test_forced_rebuild_without_toolchain_preserves_the_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    missing: str,
) -> None:
    """Forcing bypasses the freshness check, not a missing toolchain.

    An upgrade must not be aborted over a frontend the user may not even
    use while a serviceable bundle sits on disk — it warns and keeps it.
    """
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    _run_build(module, root, monkeypatch)
    bundle = root / "omnigent" / "server" / "static" / "web-ui" / "index.html"
    before = bundle.read_text()
    invocations = _run_build(
        module,
        root,
        monkeypatch,
        node=missing != "node",
        pnpm=missing != "pnpm",
        env={"OMNIGENT_BUILD_WEB_UI": "1"},
    )
    assert invocations == []
    assert bundle.read_text() == before
    assert "THE WEB UI BUNDLE IS STALE AND WAS NOT REBUILT" in capsys.readouterr().err


def test_build_failure_leaves_no_stamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stamp must only ever describe a bundle that was actually produced."""
    module = _load_setup_module()
    root = _make_project(tmp_path)
    for name in _BUILD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(module, "_project_root", lambda: root)
    monkeypatch.setattr(module, "_git_sha", lambda: _FAKE_SHA)
    monkeypatch.setattr(module, "_git_listed_files", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "_require_supported_node", lambda: "22.20.0")
    monkeypatch.setattr(module, "_resolve_pnpm_command", lambda: ["pnpm"])
    monkeypatch.setattr(
        module.subprocess,
        "run",
        mock.Mock(side_effect=subprocess.CalledProcessError(1, ["pnpm", "install"])),
    )
    with pytest.raises(SystemExit):
        module._GenerateBuildInfo._build_web_ui(object())
    assert not _stamp_path(root).is_file()


def test_fingerprint_is_content_addressed_not_mtime_addressed(tmp_path: Path) -> None:
    """Git checkouts rewrite mtimes; the fingerprint must not notice."""
    module = _load_setup_module()
    root = _make_project(tmp_path)
    with mock.patch.object(module, "_git_listed_files", return_value=None):
        before = module._web_ui_source_fingerprint(root)
        source = root / "web" / "src" / "main.tsx"
        os.utime(source, (0, 0))
        assert module._web_ui_source_fingerprint(root) == before
        source.write_text("export const app = 99;\n")
        assert module._web_ui_source_fingerprint(root) != before


def test_walk_prunes_dependency_directories(tmp_path: Path) -> None:
    module = _load_setup_module()
    root = _make_project(tmp_path)
    (root / "web" / "node_modules" / "left-pad").mkdir(parents=True)
    (root / "web" / "node_modules" / "left-pad" / "index.js").write_text("//\n")
    (root / "web" / "dist" / "assets").mkdir(parents=True)
    (root / "web" / "dist" / "assets" / "app.js").write_text("//\n")
    # web/electron/build is tracked source, so "build" must not be pruned.
    (root / "web" / "electron" / "build").mkdir(parents=True)
    (root / "web" / "electron" / "build" / "afterPack.js").write_text("//\n")
    walked = module._walk_web_ui_sources(root / "web")
    assert not [path for path in walked if "node_modules" in path.parts]
    assert not [path for path in walked if "dist" in path.parts]
    assert root / "web" / "src" / "main.tsx" in walked
    assert root / "web" / "electron" / "build" / "afterPack.js" in walked


def test_git_listing_excludes_ignored_files(tmp_path: Path) -> None:
    """The preferred enumeration defers to .gitignore for what counts as source."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    module = _load_setup_module()
    root = _make_project(tmp_path)
    (root / ".gitignore").write_text("web/node_modules/\n")
    (root / "web" / "node_modules").mkdir()
    (root / "web" / "node_modules" / "dep.js").write_text("//\n")
    _git_init(root)
    # Created only AFTER staging, so it is genuinely untracked and actually
    # tests the ``--others --exclude-standard`` half of the listing.
    (root / "web" / "src" / "untracked.tsx").write_text("export const x = 1;\n")
    assert (
        "untracked.tsx"
        in subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    listed = module._git_listed_files(root, "web")
    assert listed is not None
    names = {path.name for path in listed}
    assert {"main.tsx", "untracked.tsx"} <= names
    assert "dep.js" not in names


def test_git_listing_returns_none_outside_a_work_tree(tmp_path: Path) -> None:
    module = _load_setup_module()
    with mock.patch.object(
        module.subprocess,
        "run",
        side_effect=subprocess.CalledProcessError(128, ["git", "ls-files"]),
    ):
        assert module._git_listed_files(tmp_path, "web") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "auto"),
        ("", "auto"),
        ("  ", "auto"),
        ("maybe", "auto"),
        ("1", "force"),
        (" TRUE ", "force"),
        ("always", "force"),
        ("0", "never"),
        ("never", "never"),
    ],
)
def test_build_mode_parsing(raw: str | None, expected: str) -> None:
    module = _load_setup_module()
    assert module._web_ui_build_mode(raw) == expected


# --- Fingerprint completeness ------------------------------------------------
#
# The fingerprint defines what counts as a build input. If it can miss one,
# it produces a false SKIP — precisely the bug this mechanism exists to stop.
# Git's ignore rules are not that definition, so these run with the real git
# union live (``real_git=True``) and still demand a rebuild.


@pytest.mark.parametrize("real_git", [False, True])
def test_gitignored_build_input_change_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_git: bool
) -> None:
    """A gitignored file can still be load-bearing; Vite inlines ``.env*``."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    env_file = root / "web" / ".env.local"
    env_file.write_text("VITE_BRAIN_PICKER=1\n")
    _git_init(root, ignore="web/.env.local\nomnigent/server/static/\n")
    # git does not see it at all — only an ignore-agnostic walk does.
    assert module._git_listed_files(root, "web") is not None
    assert env_file not in (module._git_listed_files(root, "web") or [])

    assert _run_build(module, root, monkeypatch, real_git=real_git)
    env_file.write_text("VITE_BRAIN_PICKER=0\n")
    assert _run_build(module, root, monkeypatch, real_git=real_git), (
        "a changed but gitignored build input must not be a silent SKIP"
    )


@pytest.mark.parametrize("real_git", [False, True])
def test_symlinked_source_dir_change_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_git: bool
) -> None:
    """``git ls-files`` reports a dir symlink as one entry, not as its tree."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "util.ts").write_text("export const shared = 1;\n")
    (root / "web" / "src" / "shared").symlink_to(shared, target_is_directory=True)
    _git_init(root, ignore="omnigent/server/static/\n")

    assert _run_build(module, root, monkeypatch, real_git=real_git)
    (shared / "util.ts").write_text("export const shared = 2;\n")
    assert _run_build(module, root, monkeypatch, real_git=real_git), (
        "an edit inside a symlinked source tree must not be a silent SKIP"
    )


def test_real_git_enumeration_drives_the_build_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SKIP/REBUILD decision holds with the git union live, not stubbed."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    _git_init(root, ignore="omnigent/server/static/\n")
    assert _run_build(module, root, monkeypatch, real_git=True)
    assert _run_build(module, root, monkeypatch, real_git=True) == []
    (root / "web" / "src" / "main.tsx").write_text("export const app = 3;\n")
    assert _run_build(module, root, monkeypatch, real_git=True)


def test_walk_follows_symlinked_dirs_and_survives_cycles(tmp_path: Path) -> None:
    module = _load_setup_module()
    root = _make_project(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "util.ts").write_text("export const shared = 1;\n")
    (root / "web" / "src" / "shared").symlink_to(shared, target_is_directory=True)
    # A link back to an ancestor would loop forever without cycle protection.
    (root / "web" / "src" / "loop").symlink_to(root / "web", target_is_directory=True)
    walked = module._walk_web_ui_sources(root / "web")
    assert root / "web" / "src" / "shared" / "util.ts" in walked
    # Reported as traversed, not resolved, so the digest stays portable.
    assert shared / "util.ts" not in walked


def test_vite_failure_never_stamps_damaged_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Vite failure after partial output must not certify the bundle.

    Vite writes into the live bundle dir, so a mid-build failure can leave a
    half-updated SPA on disk next to the previous stamp. That stamp must
    still describe the OLD sources, or the next install would SKIP over a
    damaged bundle.
    """
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    _run_build(module, root, monkeypatch)
    stamped_sources = json.loads(_stamp_path(root).read_text())["sources"]

    (root / "web" / "src" / "main.tsx").write_text("export const broken = ;\n")
    changed_sources = module._web_ui_source_fingerprint(root)
    assert changed_sources != stamped_sources

    for name in _BUILD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(module, "_project_root", lambda: root)
    monkeypatch.setattr(module, "_git_sha", lambda: _FAKE_SHA)
    monkeypatch.setattr(module, "_git_listed_files", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "_require_supported_node", lambda: "22.20.0")
    monkeypatch.setattr(module, "_resolve_pnpm_command", lambda: ["pnpm"])

    def _install_ok_vite_fails(cmd: list[str], **_kwargs: object) -> mock.Mock:
        if "run" in cmd and "build" in cmd:
            # Simulate Vite emitting some chunks before dying.
            bundle_dir = root / "omnigent" / "server" / "static" / "web-ui"
            (bundle_dir / "assets").mkdir(exist_ok=True)
            (bundle_dir / "assets" / "partial-abc123.js").write_text("//\n")
            raise subprocess.CalledProcessError(
                1, cmd, stderr="vite: Transform failed with 1 error"
            )
        return mock.Mock(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", _install_ok_vite_fails)
    with pytest.raises(SystemExit) as excinfo:
        module._GenerateBuildInfo._build_web_ui(object())
    assert "web UI build failed" in str(excinfo.value)
    assert json.loads(_stamp_path(root).read_text())["sources"] == stamped_sources


def test_stamp_write_is_atomic_and_leaves_no_temp_files(tmp_path: Path) -> None:
    """A reader must never observe a truncated stamp."""
    module = _load_setup_module()
    target = tmp_path / "web-ui" / "omnigent-build-stamp.json"
    module._write_web_ui_stamp(target, sources="a" * 64, commit="b" * 40)
    module._write_web_ui_stamp(target, sources="c" * 64, commit="d" * 40)
    assert json.loads(target.read_text())["sources"] == "c" * 64
    assert [path.name for path in target.parent.iterdir()] == [target.name]


def test_manual_entry_point_overrides_the_ci_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Someone who typed the build command meant it, whatever the shell exports."""
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    _run_build(module, root, monkeypatch)
    monkeypatch.setenv("OMNIGENT_SKIP_WEB_UI", "true")
    monkeypatch.setattr(module, "_git_sha", lambda: _FAKE_SHA)
    monkeypatch.setattr(module, "_git_listed_files", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "_require_supported_node", lambda: "22.20.0")
    monkeypatch.setattr(module, "_resolve_pnpm_command", lambda: ["pnpm"])
    invocations: list[list[str]] = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, **_kw: (invocations.append(list(cmd)), mock.Mock(returncode=0))[1],
    )
    # The install path honours the CI skip...
    module._build_and_stamp_web_ui(root)
    assert invocations == []
    # ...the manual entry point does not, and forces past a matching stamp.
    module._build_and_stamp_web_ui(root, mode="force", respect_ci_skip=False)
    assert [cmd[1] for cmd in invocations] == ["install", "--filter"]
