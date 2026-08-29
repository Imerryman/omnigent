// Lightweight wiring guard for saved API-mount normalization.
//
//   The CLI saves server: <ws>/api/2.0/omnigent in its config. When that value
//   ends up as the desktop server_url, createWindow() must translate it to the
//   browser-facing workspace UI mount before loading the URL.
//
// URL behavior is covered in url.test.js; actual Electron boot and JSON-401
// recovery are covered by e2e/desktop_api_url_recovery.e2e.js.

"use strict";

const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");

const mainSource = readFileSync(path.join(__dirname, "../src/main.js"), "utf8");

// Strip block comments, then line comments (leaving `://` in URLs intact).
const liveCode = mainSource.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/.*$/gm, "$1");

describe("boot-time API-mount normalization", () => {
  it("imports normalizeSavedServerUrl from url.js", () => {
    assert.match(
      mainSource,
      /normalizeSavedServerUrl/,
      [
        "main.js does not import or reference normalizeSavedServerUrl.",
        "A saved server_url of .../api/2.0/omnigent is loaded raw at boot,",
        "returning a dead-end 401. Normalize the saved URL before loadURL.",
      ].join(" "),
    );
  });

  it("normalizes the saved server_url in createWindow", () => {
    // Guard that the call appears in live code (not just a comment).
    assert.match(
      liveCode,
      /normalizeSavedServerUrl\s*\(\s*loadSettings\s*\(\s*\)\s*\.\s*server_url\s*\)/,
      [
        "createWindow does not call normalizeSavedServerUrl on the saved",
        "server_url. Without this, a URL saved as .../api/2.0/omnigent is",
        "loaded directly, returning a dead-end 401 JSON page with no recovery.",
      ].join(" "),
    );
  });
});
