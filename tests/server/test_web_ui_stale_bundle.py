"""Startup warning for a web UI bundle older than the installed package.

An upgrade regenerates ``omnigent/_build_info.py`` unconditionally but only
rebuilds the SPA when its sources changed, so the server can end up serving
a months-old frontend while truthfully reporting the new version. The mount
in ``create_app`` compares the bundle's build stamp against the installed
commit and says so out loud when they disagree.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi import FastAPI

from omnigent.server import app as app_module

_INSTALLED_SHA = "0bcc734a3" + "f" * 31
_OLD_SHA = "deadbeef" + "0" * 32


@pytest.fixture(autouse=True)
def _clear_stale_check_cache() -> None:
    """Drop the per-process memoization so each case starts cold."""
    app_module._warn_if_web_ui_bundle_stale.cache_clear()


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """A mounted-bundle directory with no build stamp yet."""
    dist = tmp_path / "web-ui"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html>\n")
    return dist


def _write_stamp(dist: Path, commit: str) -> None:
    (dist / app_module._WEB_UI_BUILD_STAMP_NAME).write_text(
        json.dumps({"stamp_version": 1, "commit": commit, "sources": "x"})
    )


def _fake_build_info(monkeypatch: pytest.MonkeyPatch, value: tuple[float, str] | None) -> None:
    monkeypatch.setattr("omnigent.update_check._read_build_info", lambda: value, raising=True)


def test_matching_stamp_is_silent(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _fake_build_info(monkeypatch, (0.0, _INSTALLED_SHA))
    _write_stamp(bundle, _INSTALLED_SHA)
    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        app_module._warn_if_web_ui_bundle_stale(bundle)
    assert caplog.records == []


def test_mismatched_stamp_warns_with_both_commits(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _fake_build_info(monkeypatch, (0.0, _INSTALLED_SHA))
    _write_stamp(bundle, _OLD_SHA)
    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        app_module._warn_if_web_ui_bundle_stale(bundle)
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "STALE BUNDLE" in message
    assert _OLD_SHA[:12] in message
    assert _INSTALLED_SHA[:12] in message
    assert "pnpm --filter web run build" in message
    assert "OMNIGENT_BUILD_WEB_UI=1" in message


def test_missing_stamp_warns(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _fake_build_info(monkeypatch, (0.0, _INSTALLED_SHA))
    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        app_module._warn_if_web_ui_bundle_stale(bundle)
    assert len(caplog.records) == 1
    assert "no build stamp" in caplog.records[0].getMessage()


def test_corrupt_stamp_warns(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _fake_build_info(monkeypatch, (0.0, _INSTALLED_SHA))
    (bundle / app_module._WEB_UI_BUILD_STAMP_NAME).write_text("{not json")
    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        app_module._warn_if_web_ui_bundle_stale(bundle)
    assert len(caplog.records) == 1
    assert "no build stamp" in caplog.records[0].getMessage()


def test_unbuilt_source_checkout_is_silent(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No ``_build_info`` means nothing was installed — nothing to compare."""
    _fake_build_info(monkeypatch, None)
    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        app_module._warn_if_web_ui_bundle_stale(bundle)
    assert caplog.records == []


def test_build_without_git_is_silent(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An sdist built with no ``.git`` has no SHA to compare against."""
    _fake_build_info(monkeypatch, (0.0, ""))
    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        app_module._warn_if_web_ui_bundle_stale(bundle)
    assert caplog.records == []


def test_check_is_memoized_per_bundle_dir(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Repeat ``create_app()`` calls must not re-read the stamp or re-warn."""
    _fake_build_info(monkeypatch, (0.0, _INSTALLED_SHA))
    _write_stamp(bundle, _OLD_SHA)
    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        app_module._warn_if_web_ui_bundle_stale(bundle)
        app_module._warn_if_web_ui_bundle_stale(bundle)
    assert len(caplog.records) == 1


@pytest.fixture
def stale_web_ui_dist(bundle: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the SPA mount at a bundle built from an older commit."""
    _fake_build_info(monkeypatch, (0.0, _INSTALLED_SHA))
    _write_stamp(bundle, _OLD_SHA)
    monkeypatch.setattr(app_module, "_WEB_UI_DIST", bundle)
    return bundle


def test_stale_bundle_still_mounts_the_spa(stale_web_ui_dist: Path, app: FastAPI) -> None:
    """The check is advisory: a stale bundle warns but still mounts and serves."""
    assert any(getattr(route, "name", None) == "web-ui" for route in app.routes)


def test_externally_supplied_bundle_without_stamp_is_silent(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A deploy that ships the SPA outside the wheel has no stamp to match."""
    _fake_build_info(monkeypatch, (0.0, _INSTALLED_SHA))
    monkeypatch.setenv("OMNIGENT_WEB_UI_DIST", str(bundle))
    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        app_module._warn_if_web_ui_bundle_stale(bundle)
    assert caplog.records == []


def test_externally_supplied_bundle_with_old_stamp_still_warns(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An external bundle that does carry a stamp is still held to it."""
    _fake_build_info(monkeypatch, (0.0, _INSTALLED_SHA))
    monkeypatch.setenv("OMNIGENT_WEB_UI_DIST", str(bundle))
    _write_stamp(bundle, _OLD_SHA)
    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        app_module._warn_if_web_ui_bundle_stale(bundle)
    assert len(caplog.records) == 1
    assert "STALE BUNDLE" in caplog.records[0].getMessage()
