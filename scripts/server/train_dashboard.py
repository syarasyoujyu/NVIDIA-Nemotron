from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[2]
SFT_ROOT = ROOT / "data" / "training" / "sft"
PROBLEMS_PATH = ROOT / "data" / "problems.jsonl"


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_problem_meta() -> dict[str, dict]:
    if not PROBLEMS_PATH.exists():
        return {}
    return {row["id"]: row for row in _read_jsonl(PROBLEMS_PATH)}


def _safe_float(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def list_runs() -> list[dict]:
    runs = []
    if not SFT_ROOT.exists():
        return runs
    for path in sorted(SFT_ROOT.iterdir(), reverse=True):
        if not path.is_dir() or not (path / "config.json").exists():
            continue
        config = _read_json(path / "config.json")
        stats = config.get("stats", {})
        runs.append(
            {
                "name": path.name,
                "path": str(path.relative_to(ROOT)),
                "model": config.get("model_name", ""),
                "loss": config.get("loss_config", {}).get("name", ""),
                "epochs": config.get("num_epochs"),
                "examples": stats.get("num_examples"),
                "steps": stats.get("total_steps"),
                "unmaskedTokens": stats.get("total_unmasked_tokens"),
            }
        )
    return runs


def _flatten_epoch_loss(loss_rows: list[dict]) -> list[dict]:
    flattened = []
    for row in loss_rows:
        out = {"epoch": row.get("epoch")}
        for group in row.get("metrics", []):
            for item in group:
                out.update(item)
        flattened.append(out)
    return flattened


def _metric_keys(metrics_rows: list[dict], loss_rows: list[dict]) -> list[str]:
    keys = set()
    for row in metrics_rows:
        for key, value in row.items():
            if key not in {"epoch", "step", "time"} and _safe_float(value) is not None:
                keys.add(key)
    for row in loss_rows:
        for key, value in row.items():
            if key != "epoch" and _safe_float(value) is not None:
                keys.add(f"epoch/{key}")
    return sorted(keys)


def _latest_problem_rows(index_rows: list[dict]) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    for row in index_rows:
        key = (row.get("problem_id", ""), row.get("segment", ""))
        prev = latest.get(key)
        if prev is None or (row.get("epoch", -1), row.get("step", -1)) >= (
            prev.get("epoch", -1),
            prev.get("step", -1),
        ):
            latest[key] = row
    return list(latest.values())


def _category_summary(problem_rows: list[dict], problem_meta: dict[str, dict]) -> list[dict]:
    buckets: dict[str, dict] = defaultdict(
        lambda: {
            "category": "",
            "count": 0,
            "ruleFound": 0,
            "unknown": 0,
            "tokens": 0,
            "totalLoss": 0.0,
            "minLogprob": None,
            "minLogprobSum": 0.0,
        }
    )
    for row in problem_rows:
        category = row.get("category") or "unknown"
        bucket = buckets[category]
        bucket["category"] = category
        bucket["count"] += 1
        pid = row.get("problem_id", "")
        status = problem_meta.get(pid, {}).get("status", "")
        if status == "rule_found":
            bucket["ruleFound"] += 1
        else:
            bucket["unknown"] += 1
        tokens = int(row.get("num_loss_tokens") or 0)
        total_loss = float(row.get("total_loss") or 0.0)
        min_lp = float(row.get("min_logprob") or 0.0)
        bucket["tokens"] += tokens
        bucket["totalLoss"] += total_loss
        bucket["minLogprobSum"] += min_lp
        if bucket["minLogprob"] is None or min_lp < bucket["minLogprob"]:
            bucket["minLogprob"] = min_lp

    out = []
    for bucket in buckets.values():
        count = bucket["count"]
        tokens = bucket["tokens"]
        out.append(
            {
                "category": bucket["category"],
                "count": count,
                "ruleFound": bucket["ruleFound"],
                "unknown": bucket["unknown"],
                "ruleFoundRate": bucket["ruleFound"] / count if count else 0.0,
                "lossPerToken": bucket["totalLoss"] / tokens if tokens else 0.0,
                "minLogprob": bucket["minLogprob"] or 0.0,
                "avgMinLogprob": bucket["minLogprobSum"] / count if count else 0.0,
                "tokens": tokens,
            }
        )
    return sorted(out, key=lambda x: x["category"])


def load_run(name: str) -> dict:
    run_dir = (SFT_ROOT / name).resolve()
    if not str(run_dir).startswith(str(SFT_ROOT.resolve())):
        raise ValueError("Invalid run name")
    if not (run_dir / "config.json").exists():
        raise FileNotFoundError(f"Run not found: {name}")

    config = _read_json(run_dir / "config.json")
    metrics_rows = _read_jsonl(run_dir / "metrics.jsonl")
    epoch_loss_rows = _flatten_epoch_loss(_read_jsonl(run_dir / "loss.jsonl"))
    index_rows = _read_jsonl(run_dir / "logprobs" / "index.jsonl")
    latest_rows = _latest_problem_rows(index_rows)
    problem_meta = _load_problem_meta()
    categories = _category_summary(latest_rows, problem_meta)

    joined_rows = []
    for row in latest_rows:
        meta = problem_meta.get(row.get("problem_id", ""), {})
        tokens = int(row.get("num_loss_tokens") or 0)
        total_loss = float(row.get("total_loss") or 0.0)
        joined_rows.append(
            {
                "problemId": row.get("problem_id"),
                "segment": row.get("segment"),
                "category": row.get("category"),
                "epoch": row.get("epoch"),
                "step": row.get("step"),
                "tokens": tokens,
                "lossPerToken": total_loss / tokens if tokens else 0.0,
                "totalLoss": total_loss,
                "minLogprob": row.get("min_logprob"),
                "status": meta.get("status", ""),
                "submission": meta.get("submission", ""),
            }
        )
    joined_rows.sort(key=lambda x: (x["lossPerToken"], -float(x["minLogprob"] or 0)), reverse=True)

    total_count = len(joined_rows)
    rule_found = sum(1 for row in joined_rows if row["status"] == "rule_found")
    total_tokens = sum(row["tokens"] for row in joined_rows)
    total_loss = sum(row["totalLoss"] for row in joined_rows)
    min_logprob = min((float(row["minLogprob"] or 0.0) for row in joined_rows), default=0.0)
    last_metric = metrics_rows[-1] if metrics_rows else {}

    return {
        "name": name,
        "config": config,
        "summary": {
            "problems": total_count,
            "ruleFound": rule_found,
            "ruleFoundRate": rule_found / total_count if total_count else 0.0,
            "tokens": total_tokens,
            "lossPerToken": total_loss / total_tokens if total_tokens else 0.0,
            "minLogprob": min_logprob,
            "steps": len(metrics_rows),
            "epochs": config.get("num_epochs"),
            "lastStepLoss": last_metric.get("_loss_per_token"),
            "lastLr": last_metric.get("lr"),
        },
        "metricKeys": _metric_keys(metrics_rows, epoch_loss_rows),
        "metrics": metrics_rows,
        "epochLoss": epoch_loss_rows,
        "categories": categories,
        "problems": joined_rows[:500],
    }


INDEX_HTML = r"""
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SFT Training Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d9dee7;
      --text: #18202f;
      --muted: #687386;
      --blue: #2463eb;
      --green: #14865f;
      --red: #bf3b3b;
      --amber: #a76a00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      letter-spacing: 0;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 20px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { margin: 0; font-size: 20px; font-weight: 700; }
    select, input {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 0 10px;
      color: var(--text);
    }
    main { padding: 18px 20px 28px; max-width: 1500px; margin: 0 auto; }
    .controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .grid { display: grid; grid-template-columns: repeat(6, minmax(150px, 1fr)); gap: 12px; }
    .stat, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(20, 30, 50, 0.04);
    }
    .stat { padding: 12px; min-height: 80px; }
    .label { color: var(--muted); font-size: 12px; }
    .value { margin-top: 6px; font-size: 24px; font-weight: 700; white-space: nowrap; }
    .sub { margin-top: 4px; color: var(--muted); font-size: 12px; }
    .section { margin-top: 16px; }
    .panel { padding: 14px; }
    .panel-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 10px; }
    h2 { margin: 0; font-size: 16px; }
    .two { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(360px, .65fr); gap: 12px; }
    .chart-wrap { width: 100%; height: 330px; }
    svg { display: block; width: 100%; height: 100%; }
    .axis { stroke: #9aa4b5; stroke-width: 1; }
    .grid-line { stroke: #e8ebf0; stroke-width: 1; }
    .series { fill: none; stroke: var(--blue); stroke-width: 2.2; }
    .bar { fill: #2463eb; }
    .bar2 { fill: #15a06c; }
    table { width: 100%; border-collapse: collapse; table-layout: fixed; }
    th, td { border-bottom: 1px solid #edf0f4; padding: 8px 8px; text-align: left; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    th { color: var(--muted); font-weight: 600; font-size: 12px; background: #fbfcfe; position: sticky; top: 67px; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .good { color: var(--green); }
    .bad { color: var(--red); }
    .warn { color: var(--amber); }
    .note { color: var(--muted); line-height: 1.5; }
    .table-wrap { max-height: 520px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
    @media (max-width: 1050px) {
      .grid { grid-template-columns: repeat(2, minmax(150px, 1fr)); }
      .two { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <h1>SFT Training Dashboard</h1>
    <div class="controls">
      <label class="label">Run</label>
      <select id="runSelect"></select>
      <label class="label">Metric</label>
      <select id="metricSelect"></select>
    </div>
  </header>
  <main>
    <div class="grid" id="stats"></div>
    <section class="section two">
      <div class="panel">
        <div class="panel-head">
          <h2>Metric / Loss Trend</h2>
          <span class="note" id="trendNote"></span>
        </div>
        <div class="chart-wrap"><svg id="lineChart"></svg></div>
      </div>
      <div class="panel">
        <div class="panel-head"><h2>Category Accuracy</h2></div>
        <div class="chart-wrap"><svg id="barChart"></svg></div>
      </div>
    </section>
    <section class="section panel">
      <div class="panel-head">
        <h2>Category Summary</h2>
        <span class="note">Accuracy here means status == rule_found in data/problems.jsonl.</span>
      </div>
      <div class="table-wrap"><table id="categoryTable"></table></div>
    </section>
    <section class="section panel">
      <div class="panel-head">
        <h2>Worst Problem Rows</h2>
        <input id="problemFilter" placeholder="filter problem/category/status">
      </div>
      <div class="table-wrap"><table id="problemTable"></table></div>
    </section>
  </main>
  <script>
    const fmt = new Intl.NumberFormat("en-US");
    let state = { runs: [], run: null };

    const $ = (id) => document.getElementById(id);
    const pct = (v) => `${(100 * (Number(v) || 0)).toFixed(1)}%`;
    const num = (v, digits = 4) => Number.isFinite(Number(v)) ? Number(v).toFixed(digits) : "";

    async function getJson(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    function stat(label, value, sub = "") {
      return `<div class="stat"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
    }

    function renderStats(run) {
      const s = run.summary;
      $("stats").innerHTML = [
        stat("Rule-found Rate", pct(s.ruleFoundRate), `${fmt.format(s.ruleFound)} / ${fmt.format(s.problems)} problems`),
        stat("Loss / Token", num(s.lossPerToken, 5), `last step ${num(s.lastStepLoss, 5)}`),
        stat("Min Logprob", num(s.minLogprob, 3), "lowest observed problem token"),
        stat("Steps", fmt.format(s.steps), `${s.epochs ?? ""} epoch(s)`),
        stat("Tokens", fmt.format(s.tokens), "unmasked loss tokens"),
        stat("Learning Rate", num(s.lastLr, 8), "last recorded step"),
      ].join("");
    }

    function metricSeries(run, key) {
      if (key.startsWith("epoch/")) {
        const k = key.slice(6);
        return run.epochLoss.map((row) => ({ x: row.epoch, y: row[k] })).filter((p) => Number.isFinite(Number(p.y)));
      }
      return run.metrics.map((row) => ({ x: row.step, y: row[key] })).filter((p) => Number.isFinite(Number(p.y)));
    }

    function drawLine(svg, data) {
      svg.innerHTML = "";
      const w = svg.clientWidth || 800, h = svg.clientHeight || 320;
      const pad = { l: 54, r: 16, t: 18, b: 34 };
      if (data.length === 0) return;
      const xs = data.map(d => Number(d.x)), ys = data.map(d => Number(d.y));
      const xmin = Math.min(...xs), xmax = Math.max(...xs);
      let ymin = Math.min(...ys), ymax = Math.max(...ys);
      if (ymin === ymax) { ymin -= 1; ymax += 1; }
      const x = (v) => pad.l + (xmax === xmin ? 0 : (v - xmin) / (xmax - xmin)) * (w - pad.l - pad.r);
      const y = (v) => h - pad.b - (v - ymin) / (ymax - ymin) * (h - pad.t - pad.b);
      for (let i = 0; i <= 4; i++) {
        const yy = pad.t + i * (h - pad.t - pad.b) / 4;
        svg.insertAdjacentHTML("beforeend", `<line class="grid-line" x1="${pad.l}" y1="${yy}" x2="${w-pad.r}" y2="${yy}"/>`);
      }
      svg.insertAdjacentHTML("beforeend", `<line class="axis" x1="${pad.l}" y1="${h-pad.b}" x2="${w-pad.r}" y2="${h-pad.b}"/>`);
      svg.insertAdjacentHTML("beforeend", `<line class="axis" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${h-pad.b}"/>`);
      const points = data.map(d => `${x(Number(d.x)).toFixed(1)},${y(Number(d.y)).toFixed(1)}`).join(" ");
      svg.insertAdjacentHTML("beforeend", `<polyline class="series" points="${points}"/>`);
      svg.insertAdjacentHTML("beforeend", `<text x="${pad.l}" y="${h-8}" fill="#687386" font-size="11">step/epoch ${xmin} to ${xmax}</text>`);
      svg.insertAdjacentHTML("beforeend", `<text x="8" y="${pad.t+4}" fill="#687386" font-size="11">${num(ymax, 4)}</text>`);
      svg.insertAdjacentHTML("beforeend", `<text x="8" y="${h-pad.b}" fill="#687386" font-size="11">${num(ymin, 4)}</text>`);
    }

    function drawBars(svg, rows) {
      svg.innerHTML = "";
      const w = svg.clientWidth || 520, h = svg.clientHeight || 320;
      const pad = { l: 150, r: 16, t: 14, b: 20 };
      const data = rows.slice().sort((a, b) => b.ruleFoundRate - a.ruleFoundRate);
      const rowH = Math.max(22, (h - pad.t - pad.b) / Math.max(1, data.length));
      data.forEach((d, i) => {
        const y = pad.t + i * rowH;
        const bw = (w - pad.l - pad.r) * d.ruleFoundRate;
        svg.insertAdjacentHTML("beforeend", `<text x="8" y="${y+15}" fill="#18202f" font-size="12">${d.category}</text>`);
        svg.insertAdjacentHTML("beforeend", `<rect x="${pad.l}" y="${y+3}" width="${w-pad.l-pad.r}" height="12" fill="#edf0f4"/>`);
        svg.insertAdjacentHTML("beforeend", `<rect class="bar2" x="${pad.l}" y="${y+3}" width="${bw}" height="12"/>`);
        svg.insertAdjacentHTML("beforeend", `<text x="${w-pad.r-44}" y="${y+15}" fill="#18202f" font-size="11">${pct(d.ruleFoundRate)}</text>`);
      });
    }

    function renderTable(el, columns, rows) {
      el.innerHTML = `<thead><tr>${columns.map(c => `<th class="${c.num ? "num" : ""}">${c.label}</th>`).join("")}</tr></thead>` +
        `<tbody>${rows.map(row => `<tr>${columns.map(c => `<td class="${c.num ? "num" : ""}">${c.render ? c.render(row) : (row[c.key] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody>`;
    }

    function renderCategoryTable(rows) {
      renderTable($("categoryTable"), [
        { label: "Category", key: "category" },
        { label: "Problems", key: "count", num: true, render: r => fmt.format(r.count) },
        { label: "Rule Found", key: "ruleFound", num: true, render: r => fmt.format(r.ruleFound) },
        { label: "Rate", key: "ruleFoundRate", num: true, render: r => pct(r.ruleFoundRate) },
        { label: "Loss / Token", key: "lossPerToken", num: true, render: r => num(r.lossPerToken, 5) },
        { label: "Worst Min LP", key: "minLogprob", num: true, render: r => num(r.minLogprob, 3) },
        { label: "Tokens", key: "tokens", num: true, render: r => fmt.format(r.tokens) },
      ], rows);
    }

    function renderProblemTable() {
      const q = $("problemFilter").value.trim().toLowerCase();
      const rows = state.run.problems.filter(r => !q || [r.problemId, r.category, r.status].join(" ").toLowerCase().includes(q)).slice(0, 200);
      renderTable($("problemTable"), [
        { label: "Problem", key: "problemId" },
        { label: "Category", key: "category" },
        { label: "Status", key: "status", render: r => `<span class="${r.status === "rule_found" ? "good" : "warn"}">${r.status}</span>` },
        { label: "Step", key: "step", num: true },
        { label: "Loss / Token", key: "lossPerToken", num: true, render: r => num(r.lossPerToken, 5) },
        { label: "Min Logprob", key: "minLogprob", num: true, render: r => num(r.minLogprob, 3) },
        { label: "Tokens", key: "tokens", num: true, render: r => fmt.format(r.tokens) },
        { label: "Submission", key: "submission" },
      ], rows);
    }

    function renderRun(run) {
      state.run = run;
      renderStats(run);
      const metricSelect = $("metricSelect");
      const preferred = ["_loss_per_token", "fwd/loss:sum", "epoch/nll_per_token"];
      metricSelect.innerHTML = run.metricKeys.map(k => `<option value="${k}">${k}</option>`).join("");
      metricSelect.value = preferred.find(k => run.metricKeys.includes(k)) || run.metricKeys[0] || "";
      renderCharts();
      renderCategoryTable(run.categories);
      renderProblemTable();
    }

    function renderCharts() {
      const key = $("metricSelect").value;
      $("trendNote").textContent = key;
      drawLine($("lineChart"), metricSeries(state.run, key));
      drawBars($("barChart"), state.run.categories);
    }

    async function loadRun(name) {
      renderRun(await getJson(`/api/run?name=${encodeURIComponent(name)}`));
    }

    async function init() {
      state.runs = await getJson("/api/runs");
      $("runSelect").innerHTML = state.runs.map(r => `<option value="${r.name}">${r.name}</option>`).join("");
      $("runSelect").addEventListener("change", e => loadRun(e.target.value));
      $("metricSelect").addEventListener("change", renderCharts);
      $("problemFilter").addEventListener("input", renderProblemTable);
      if (state.runs.length) await loadRun(state.runs[0].name);
    }
    init().catch(err => {
      document.body.innerHTML = `<pre style="padding:20px;color:#bf3b3b">${err.stack || err}</pre>`;
    });
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: object, status: int = 200) -> None:
        self._send(
            status,
            json.dumps(data, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/runs":
                self._json(list_runs())
            elif parsed.path == "/api/run":
                params = parse_qs(parsed.query)
                name = params.get("name", [""])[0]
                self._json(load_run(name))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve SFT training dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving training dashboard at {url}")
    server.serve_forever()


if __name__ == "__main__":
    main()
