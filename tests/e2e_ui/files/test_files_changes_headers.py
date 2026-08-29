"""E2E coverage for the Files and Changes panel headers."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import Page, Route, expect

from tests.e2e_ui.conftest import (
    _build_hello_world_bundle,
    _server_state,
    open_right_rail,
)


@pytest.fixture
def empty_workspace_session(
    live_server: str,
    tmp_path: Path,
) -> Iterator[tuple[str, str]]:
    """Create a runner-bound session whose real workspace has no changes."""
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    response = httpx.post(
        f"{live_server}/v1/sessions",
        data={"metadata": json.dumps({"workspace": str(tmp_path)})},
        files={"bundle": ("agent.tar.gz", _build_hello_world_bundle(), "application/gzip")},
        timeout=30.0,
    )
    response.raise_for_status()
    session_id = response.json()["session_id"]
    bind = httpx.patch(
        f"{live_server}/v1/sessions/{session_id}",
        json={"runner_id": str(_server_state["runner_id"])},
        timeout=10.0,
    )
    bind.raise_for_status()

    try:
        yield live_server, session_id
    finally:
        httpx.delete(f"{live_server}/v1/sessions/{session_id}", timeout=10.0)


def _capture_screenshot(page: Page, name: str) -> None:
    screenshot_dir = os.environ.get("E2E_SCREENSHOT_DIR")
    if not screenshot_dir:
        return
    destination = Path(screenshot_dir)
    destination.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(destination / f"{name}.png"))


def test_files_and_changes_tabs_update_the_visible_heading(
    page: Page,
    empty_workspace_session: tuple[str, str],
) -> None:
    """Switching tabs updates the selected tab, heading, and empty-state body."""
    base_url, session_id = empty_workspace_session
    page.goto(f"{base_url}/c/{session_id}")
    open_right_rail(page)

    rail = page.get_by_role("complementary", name="Workspace")
    content = rail.locator("[data-workspace-panel-content]")
    files_tab = rail.get_by_role("tab", name="Files", exact=True)
    changes_tab = rail.get_by_role("tab", name="Changes", exact=True)

    files_tab.click()
    expect(files_tab).to_have_attribute("aria-selected", "true")
    expect(changes_tab).to_have_attribute("aria-selected", "false")
    expect(content.get_by_role("heading", name="Working folder", exact=True)).to_be_visible()
    expect(content.get_by_role("heading", name="Changes", exact=True)).to_have_count(0)
    expect(content.get_by_role("searchbox", name="Search all files")).to_be_visible()

    changes_tab.click()
    expect(files_tab).to_have_attribute("aria-selected", "false")
    expect(changes_tab).to_have_attribute("aria-selected", "true")
    expect(content.get_by_role("heading", name="Changes", exact=True)).to_be_visible()
    expect(content.get_by_role("heading", name="Working folder", exact=True)).to_have_count(0)
    expect(content.get_by_text("No workspace changes yet", exact=True)).to_be_visible(
        timeout=30_000
    )
    _capture_screenshot(page, "changes-tab-empty-header")

    files_tab.click()
    expect(content.get_by_role("heading", name="Working folder", exact=True)).to_be_visible()
    expect(content.get_by_role("heading", name="Changes", exact=True)).to_have_count(0)


def test_changes_heading_remains_visible_while_changes_load(
    page: Page,
    empty_workspace_session: tuple[str, str],
) -> None:
    """The Changes heading identifies the panel before its request completes."""
    base_url, session_id = empty_workspace_session
    held_routes: list[Route] = []

    page.route(
        re.compile(
            rf"/v1/sessions/{re.escape(session_id)}/resources/environments/[^/]+/changes(\?|$)"
        ),
        lambda route: held_routes.append(route),
        times=1,
    )
    page.goto(f"{base_url}/c/{session_id}")
    open_right_rail(page)

    rail = page.get_by_role("complementary", name="Workspace")
    content = rail.locator("[data-workspace-panel-content]")
    rail.get_by_role("tab", name="Changes", exact=True).click()

    expect(content.get_by_role("heading", name="Changes", exact=True)).to_be_visible()
    expect(content.get_by_text("Loading…", exact=True)).to_be_visible()
    assert len(held_routes) == 1

    held_routes[0].fulfill(
        status=200,
        headers={"content-type": "application/json"},
        body=json.dumps({"object": "list", "data": [], "has_more": False}),
    )
    expect(content.get_by_text("No workspace changes yet", exact=True)).to_be_visible()
