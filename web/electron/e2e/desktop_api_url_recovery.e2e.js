// Real-Electron regression lane for saved Databricks API URLs and runtime 401s.
//
// A local HTTP server stands in for the workspace boundary while Chromium maps
// workspace-shaped hostnames to loopback. The production Electron main process,
// BrowserWindow navigation events, bundled setup page, and response handling
// all run unchanged.

"use strict";

const { after, before, describe, it } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const os = require("node:os");
const path = require("node:path");

const { desktopDepsAvailable, launchDesktop, saveRecording } = require("./desktopHarness");

const WORKSPACE_HOST = "workspace.cloud.databricks.com";
const FOREIGN_HOST = "foreign.example.com";
const RECORD_DIR = path.join(__dirname, "recordings", "desktop-api-url-recovery");
const deps = desktopDepsAvailable();

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve(server.address().port));
  });
}

function closeServer(server) {
  return new Promise((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
  });
}

function html(res, status, body) {
  res.writeHead(status, { "Content-Type": "text/html; charset=utf-8" });
  res.end(body);
}

function json401(res) {
  res.writeHead(401, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify({ error: "credential missing" }));
}

async function expectSetupFallback(window) {
  const error = window.locator("#err");
  await error.waitFor({ state: "visible", timeout: 10_000 });
  assert.match(await error.textContent(), /401 Unauthorized/);
}

describe(
  "desktop shell — Databricks API URL boot and 401 recovery",
  { skip: deps.ok ? false : `missing deps: ${deps.missing.join(", ")}` },
  () => {
    let server;
    let port;
    let requests;

    before(async () => {
      requests = [];
      server = http.createServer((req, res) => {
        const url = new URL(req.url, `http://${req.headers.host}`);
        requests.push({ host: url.hostname, path: `${url.pathname}${url.search}` });

        if (url.pathname === "/omnigent") {
          html(res, 200, '<main id="workspace-ui">Workspace UI</main>');
          return;
        }
        if (url.pathname === "/frame-host") {
          html(
            res,
            200,
            '<main id="frame-host">Frame host</main><iframe id="probe" src="/json-401"></iframe>',
          );
          return;
        }
        if (url.pathname === "/redirect-json-401") {
          res.writeHead(302, { Location: "/json-401" });
          res.end();
          return;
        }
        if (url.pathname === "/json-401") {
          json401(res);
          return;
        }
        if (url.pathname === "/html-401") {
          html(res, 401, "<main>Sign in required</main>");
          return;
        }
        html(res, 404, "<main>Not found</main>");
      });
      port = await listen(server);
    });

    after(async () => {
      if (server?.listening) await closeServer(server);
    });

    it("normalizes URL state and safely recovers a same-origin JSON 401", async () => {
      const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "omni-desktop-api-e2e-"));
      const savedApiUrl = `http://${WORKSPACE_HOST}:${port}/api/2.0/omnigent/?o=123#saved-conversation`;
      const launchArgs = [
        `--host-resolver-rules=MAP ${WORKSPACE_HOST} 127.0.0.1, MAP ${FOREIGN_HOST} 127.0.0.1`,
      ];
      let electronApp;
      let window;
      let saved;
      try {
        ({ electronApp, window } = await launchDesktop({
          recordDir: RECORD_DIR,
          serverUrl: savedApiUrl,
          userDataDir: tmpDir,
          launchArgs,
        }));

        await window.locator("#workspace-ui").waitFor({ state: "visible", timeout: 10_000 });
        assert.equal(
          window.url(),
          `http://${WORKSPACE_HOST}:${port}/omnigent?o=123#saved-conversation`,
        );
        assert.ok(
          requests.some(
            (request) => request.host === WORKSPACE_HOST && request.path === "/omnigent?o=123",
          ),
          "Electron never requested the normalized workspace UI mount",
        );
        assert.equal(
          requests.some((request) => request.path.startsWith("/api/2.0/omnigent")),
          false,
          "Electron navigated to the saved API mount",
        );

        // A JSON 401 in a child frame must not replace the top-level document.
        await window.goto(`http://${WORKSPACE_HOST}:${port}/frame-host`);
        await window
          .frameLocator("#probe")
          .locator("body")
          .waitFor({ state: "visible", timeout: 10_000 });
        assert.equal(new URL(window.url()).pathname, "/frame-host");
        assert.equal(await window.locator("#frame-host").isVisible(), true);

        // A foreign top-level JSON 401 must not be trusted merely because it
        // chose the same status and MIME type as the pinned server.
        await window.goto(`http://${FOREIGN_HOST}:${port}/json-401`);
        await window.getByText(/credential missing/).waitFor({ state: "visible", timeout: 10_000 });
        assert.equal(new URL(window.url()).hostname, FOREIGN_HOST);
        assert.equal(await window.locator("#err").count(), 0);

        // Returning to the pinned origin and following a redirect into a JSON
        // 401 exercises the real did-navigate fallback. The pin is cleared
        // before loading setup, so the response cannot create a redirect loop.
        await window.goto(`http://${WORKSPACE_HOST}:${port}/omnigent`);
        const requestCountBeforeFallback = requests.length;
        await window.goto(`http://${WORKSPACE_HOST}:${port}/redirect-json-401`).catch(() => {});
        await expectSetupFallback(window);
        assert.equal(
          await window.locator("#url").inputValue(),
          `http://${WORKSPACE_HOST}:${port}/omnigent?o=123#saved-conversation`,
        );
        const requestCountAfterFallback = requests.length;
        assert.ok(
          requestCountAfterFallback >= requestCountBeforeFallback + 2,
          "redirect and JSON-401 requests were not both observed",
        );
        await window.waitForTimeout(500);
        assert.equal(requests.length, requestCountAfterFallback, "401 fallback entered a loop");
        await expectSetupFallback(window);
        await window.screenshot({
          path: path.join(RECORD_DIR, "api-url-and-json-401.png"),
        });
      } finally {
        if (electronApp) await electronApp.close().catch(() => {});
        saved = saveRecording(RECORD_DIR, "api-url-and-json-401");
        fs.rmSync(tmpDir, { recursive: true, force: true });
      }
      assert.ok(saved && saved.length > 0, "no desktop recording was produced");
    });

    it("supports the legacy API path and current-main non-JSON fallback", async () => {
      const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "omni-desktop-legacy-e2e-"));
      const legacyApiUrl = `http://${WORKSPACE_HOST}:${port}/api/2.0/omnigents/`;
      let electronApp;
      let window;
      let saved;
      try {
        ({ electronApp, window } = await launchDesktop({
          recordDir: RECORD_DIR,
          serverUrl: legacyApiUrl,
          userDataDir: tmpDir,
          launchArgs: [`--host-resolver-rules=MAP ${WORKSPACE_HOST} 127.0.0.1`],
        }));
        await window.locator("#workspace-ui").waitFor({ state: "visible", timeout: 10_000 });
        assert.equal(window.url(), `http://${WORKSPACE_HOST}:${port}/omnigent`);

        // Current main deliberately recovers every same-origin top-level HTTP
        // error, not just JSON. Keep that broader behavior while proving the
        // PR no longer needs another session-wide response listener.
        await window.goto(`http://${WORKSPACE_HOST}:${port}/html-401`).catch(() => {});
        await expectSetupFallback(window);
        await window.screenshot({
          path: path.join(RECORD_DIR, "legacy-api-and-html-401.png"),
        });
      } finally {
        if (electronApp) await electronApp.close().catch(() => {});
        saved = saveRecording(RECORD_DIR, "legacy-api-and-html-401");
        fs.rmSync(tmpDir, { recursive: true, force: true });
      }
      assert.ok(saved && saved.length > 0, "no desktop recording was produced");
    });
  },
);
