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
    assert "setup.py --omnigent-build-web-ui" in message
    assert "OMNIGENT_BUILD_WEB_UI=1" in message


def test_missing_stamp_warns(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _fake_build_info(monkeypatch, (0.0, _INSTALLED_SHA))
    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        app_module._warn_if_web_ui_bundle_stale(bundle)
    assert len(caplog.records) == 1
    assert "no readable build stamp" in caplog.records[0].getMessage()


def test_corrupt_stamp_warns(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _fake_build_info(monkeypatch, (0.0, _INSTALLED_SHA))
    (bundle / app_module._WEB_UI_BUILD_STAMP_NAME).write_text("{not json")
    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        app_module._warn_if_web_ui_bundle_stale(bundle)
    assert len(caplog.records) == 1
    assert "no readable build stamp" in caplog.records[0].getMessage()


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


@pytest.mark.parametrize("stamp", [None, "corrupt", _OLD_SHA])
def test_externally_supplied_bundle_gets_no_exemption(
    bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stamp: str | None,
) -> None:
    """A bundle shipped outside the wheel is held to the same standard.

    Unknown provenance is exactly the condition this check exists to
    surface, so neither a missing nor an unreadable stamp buys silence
    just because ``OMNIGENT_WEB_UI_DIST`` is pointing at the bundle.
    """
    _fake_build_info(monkeypatch, (0.0, _INSTALLED_SHA))
    monkeypatch.setenv("OMNIGENT_WEB_UI_DIST", str(bundle))
    if stamp == "corrupt":
        (bundle / app_module._WEB_UI_BUILD_STAMP_NAME).write_text("{not json")
    elif stamp is not None:
        _write_stamp(bundle, stamp)
    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        app_module._warn_if_web_ui_bundle_stale(bundle)
    assert len(caplog.records) == 1


def test_stale_external_bundle_still_mounts_the_spa(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Warning about an external bundle must not stop it being served."""
    _fake_build_info(monkeypatch, (0.0, _INSTALLED_SHA))
    monkeypatch.setenv("OMNIGENT_WEB_UI_DIST", str(bundle))
    monkeypatch.setattr(app_module, "_WEB_UI_DIST", bundle)
    app = request.getfixturevalue("app")
    assert any(getattr(route, "name", None) == "web-ui" for route in app.routes)


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_explicit_opt_out_silences_the_warning(
    bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    value: str,
) -> None:
    """The only way out is the documented, default-off env var."""
    _fake_build_info(monkeypatch, (0.0, _INSTALLED_SHA))
    _write_stamp(bundle, _OLD_SHA)
    monkeypatch.setenv(app_module._WEB_UI_STALE_WARNING_OPT_OUT, value)
    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        app_module._warn_if_web_ui_bundle_stale(bundle)
    assert caplog.records == []


@pytest.mark.parametrize("value", ["", "0", "false", "off", "nonsense"])
def test_opt_out_is_off_by_default_and_on_junk(
    bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    value: str,
) -> None:
    _fake_build_info(monkeypatch, (0.0, _INSTALLED_SHA))
    _write_stamp(bundle, _OLD_SHA)
    monkeypatch.setenv(app_module._WEB_UI_STALE_WARNING_OPT_OUT, value)
    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        app_module._warn_if_web_ui_bundle_stale(bundle)
    assert len(caplog.records) == 1


def test_advertised_fix_points_at_a_command_that_writes_a_stamp(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A bare `pnpm run build` leaves no stamp, so it cannot be the advice."""
    _fake_build_info(monkeypatch, (0.0, _INSTALLED_SHA))
    _write_stamp(bundle, _OLD_SHA)
    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        app_module._warn_if_web_ui_bundle_stale(bundle)
    message = caplog.records[0].getMessage()
    assert "setup.py --omnigent-build-web-ui" in message
    assert "OMNIGENT_BUILD_WEB_UI=1" in message
    assert "writes no stamp" in message
