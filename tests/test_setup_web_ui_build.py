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
    # Force the pruned-walk path: the tmp dir is not a work tree, and the
    # outcome must not depend on whether it happens to sit inside one.
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

    def _fake_run(cmd: list[str], **_kwargs: object) -> mock.Mock:
        invocations.append(list(cmd))
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


def test_forced_rebuild_without_toolchain_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit force is never silently downgraded to a warning."""
    module = _load_setup_module()
    root = _make_project(tmp_path, bundle=True)
    _run_build(module, root, monkeypatch)
    with pytest.raises(SystemExit):
        _run_build(
            module,
            root,
            monkeypatch,
            node=False,
            env={"OMNIGENT_BUILD_WEB_UI": "1"},
        )


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
    (root / "web" / "src" / "untracked.tsx").write_text("export const x = 1;\n")
    for args in (["init", "-q"], ["add", "-A"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
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
