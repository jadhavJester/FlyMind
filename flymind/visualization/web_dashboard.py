"""
FlyMind Web Research Dashboard.
Lightweight HTTP web server providing an interactive browser console for real-time simulation,
neural activity inspection, and benchmark execution.
"""
import http.server
import json
import socketserver
import threading
import time
from typing import Any, Dict
import numpy as np

from ..evaluation.benchmarks.scenario_runner import BenchmarkRunner, ControllerType


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FlyMind: Insect-Inspired Flight Intelligence Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0d13;
      --card-bg: rgba(22, 27, 34, 0.75);
      --border: rgba(255, 255, 255, 0.1);
      --accent: #58a6ff;
      --accent-green: #3fb950;
      --accent-red: #f85149;
      --accent-amber: #d29922;
      --text: #f0f6fc;
      --text-muted: #8b949e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', -apple-system, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 1.25rem 2rem;
      border-bottom: 1px solid var(--border);
      background: rgba(13, 17, 23, 0.85);
      backdrop-filter: blur(12px);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .brand h1 {
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(90deg, #58a6ff, #bc8cff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .badge {
      font-size: 0.75rem;
      padding: 0.2rem 0.5rem;
      border-radius: 999px;
      background: rgba(88, 166, 255, 0.15);
      color: var(--accent);
      border: 1px solid rgba(88, 166, 255, 0.3);
      font-family: 'JetBrains Mono', monospace;
    }
    main {
      padding: 2rem;
      display: grid;
      grid-template-columns: 1fr 380px;
      gap: 1.75rem;
      max-width: 1440px;
      width: 100%;
      margin: 0 auto;
      flex: 1;
    }
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      backdrop-filter: blur(16px);
      padding: 1.5rem;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    .panel h2 {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .status-badge {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent-green);
      box-shadow: 0 0 8px var(--accent-green);
    }
    .canvas-container {
      position: relative;
      background: #000;
      border-radius: 8px;
      overflow: hidden;
      aspect-ratio: 4/3;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--border);
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .controls-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
      margin-top: 1rem;
    }
    select, button {
      background: rgba(30, 36, 46, 0.85);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.65rem 0.85rem;
      border-radius: 6px;
      font-family: inherit;
      font-size: 0.85rem;
      outline: none;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    select:hover, button:hover {
      border-color: var(--accent);
      background: rgba(40, 48, 62, 0.95);
    }
    button.primary {
      background: linear-gradient(135deg, #1f6feb, #238636);
      border: none;
      font-weight: 600;
      grid-column: span 2;
    }
    button.primary:hover {
      opacity: 0.92;
      box-shadow: 0 4px 12px rgba(35, 134, 54, 0.35);
    }
    .telemetry-row {
      display: flex;
      justify-content: space-between;
      padding: 0.5rem 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
    }
    .telemetry-label {
      color: var(--text-muted);
    }
    .metric-val {
      font-weight: 600;
      color: var(--accent);
    }
    .bar-container {
      width: 100%;
      height: 8px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      margin-top: 4px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: var(--accent-green);
      transition: width 0.15s ease;
    }
    #benchmark-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;
      font-size: 0.78rem;
    }
    #benchmark-table th, #benchmark-table td {
      padding: 0.5rem 0.4rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.07);
      text-align: left;
    }
    #benchmark-table th {
      color: var(--text-muted);
      font-weight: 600;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span class="status-badge"></span>
      <h1>FlyMind</h1>
      <span class="badge">Fruit Fly Visual Flight Intelligence</span>
    </div>
    <div style="font-size: 0.85rem; color: var(--text-muted);">
      6-DoF Dynamics &bull; LPLC2 &bull; Giant Fiber LIF &bull; WTA Gap Selector
    </div>
  </header>

  <main>
    <section class="panel">
      <h2>Simulation Visual Field & Telemetry HUD</h2>
      <div class="canvas-container">
        <canvas id="viewCanvas" width="640" height="480"></canvas>
      </div>
      <div class="controls-grid">
        <select id="ctrlSelect">
          <option value="FLYMIND_FULL">FlyMindBio (Full Insect Circuit)</option>
          <option value="LPLC2_ONLY">LPLC2-GF Escape Only</option>
          <option value="WTA_FLOW_ONLY">WTA Optical Flow Gap Only</option>
          <option value="CANNY_BASELINE">Canny Edge Density Baseline</option>
          <option value="DEPTHNET_BASELINE">Monocular DepthNet Baseline</option>
        </select>
        <select id="envSelect">
          <option value="forest">Obstacle Forest (Poisson cylinders)</option>
          <option value="gates">Narrow Slit Gates (Gap finding)</option>
          <option value="looming">Dynamic Looming Hazards</option>
        </select>
        <button class="primary" id="runBtn" onclick="runFlight()">Run Flight Episode</button>
      </div>
    </section>

    <aside class="panel" style="display: flex; flex-direction: column; gap: 1rem;">
      <h2>Neural & Flight Telemetry</h2>
      <div class="telemetry-box">
        <div class="telemetry-row">
          <span class="telemetry-label">Flight Mode</span>
          <span class="metric-val" id="modeVal">CRUISE</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Giant Fiber Vm</span>
          <span class="metric-val" id="gfVal">-65.0 mV</span>
        </div>
        <div class="bar-container">
          <div class="bar-fill" id="gfBar" style="width: 10%;"></div>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Collision Risk</span>
          <span class="metric-val" id="riskVal">0.05</span>
        </div>
        <div class="bar-container">
          <div class="bar-fill" id="riskBar" style="width: 5%;"></div>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Obstacle Clearance</span>
          <span class="metric-val" id="clearVal">4.20 m</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Forward Speed</span>
          <span class="metric-val" id="speedVal">1.5 m/s</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Safety Supervisor</span>
          <span class="metric-val" style="color: var(--accent-green);" id="safetyVal">NOMINAL</span>
        </div>
      </div>

      <div style="margin-top: auto;">
        <h2>Benchmark Status</h2>
        <div id="resultStatus" style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.4;">
          Ready to launch reproducible simulation episodes.
        </div>
      </div>
    </aside>
  </main>

  <script>
    const canvas = document.getElementById('viewCanvas');
    const ctx = canvas.getContext('2d');

    function drawIdle() {
      ctx.fillStyle = '#0f141c';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#58a6ff';
      ctx.font = '16px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Press "Run Flight Episode" to simulate', canvas.width/2, canvas.height/2);
    }
    drawIdle();

    async function runFlight() {
      const btn = document.getElementById('runBtn');
      btn.disabled = true;
      btn.innerText = 'Simulating Flight Episode...';
      document.getElementById('resultStatus').innerText = 'Simulating 6-DoF dynamics, bio-perception & neural circuits...';

      const ctrl = document.getElementById('ctrlSelect').value;
      const env = document.getElementById('envSelect').value;

      try {
        const resp = await fetch(`/api/run_episode?ctrl=${ctrl}&env=${env}&seed=${Math.floor(Math.random()*1000)}`);
        const data = await resp.json();

        document.getElementById('modeVal').innerText = data.has_collided ? 'COLLISION' : (data.reached_goal ? 'GOAL_REACHED' : 'HOVER');
        document.getElementById('clearVal').innerText = `${data.min_clearance_m.toFixed(2)} m`;
        document.getElementById('speedVal').innerText = `${data.mean_speed_m_s.toFixed(2)} m/s`;
        document.getElementById('resultStatus').innerHTML = `
          <strong>Result:</strong> ${data.reached_goal ? 'Goal Reached!' : (data.has_collided ? 'Collision Detected' : 'Time limit reached')}<br>
          <strong>Path Length:</strong> ${data.path_length_m.toFixed(1)} m<br>
          <strong>Min Clearance:</strong> ${data.min_clearance_m.toFixed(2)} m<br>
          <strong>Saccades:</strong> ${data.saccade_count}<br>
          <strong>Step Latency:</strong> ${data.mean_computation_time_ms.toFixed(2)} ms
        `;

        // Render mock trajectory visualization on canvas
        ctx.fillStyle = '#080a0f';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = data.has_collided ? '#f85149' : '#3fb950';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(30, canvas.height / 2);
        const steps = 40;
        for (let i = 1; i <= steps; i++) {
          const x = 30 + (i / steps) * (canvas.width - 60);
          const y = canvas.height / 2 + Math.sin(i * 0.4) * 45;
          ctx.lineTo(x, y);
        }
        ctx.stroke();

        ctx.fillStyle = '#f0f6fc';
        ctx.font = '14px JetBrains Mono';
        ctx.fillText(`Trajectory: ${data.controller_name} in ${data.scenario_name}`, 20, 30);
      } catch (err) {
        document.getElementById('resultStatus').innerText = 'Error: ' + err;
      } finally {
        btn.disabled = false;
        btn.innerText = 'Run Flight Episode';
      }
    }
  </script>
</body>
</html>
"""


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """Handles HTTP requests for research dashboard."""
    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if self.path.startswith("/api/run_episode"):
            # Parse query params
            from urllib.parse import parse_qs, urlparse
            query = parse_qs(urlparse(self.path).query)
            ctrl_str = query.get("ctrl", ["FLYMIND_FULL"])[0]
            env_str = query.get("env", ["forest"])[0]
            seed = int(query.get("seed", [42])[0])

            ctrl_map = {
                "FLYMIND_FULL": ControllerType.FLYMIND_FULL,
                "LPLC2_ONLY": ControllerType.LPLC2_ONLY,
                "WTA_FLOW_ONLY": ControllerType.WTA_FLOW_ONLY,
                "CANNY_BASELINE": ControllerType.CANNY_BASELINE,
                "DEPTHNET_BASELINE": ControllerType.DEPTHNET_BASELINE,
            }
            ctrl_type = ctrl_map.get(ctrl_str, ControllerType.FLYMIND_FULL)

            runner = BenchmarkRunner(dt=0.02, max_episode_time_s=6.0)
            result = runner.run_episode(controller_type=ctrl_type, environment_name=env_str, seed=seed)

            payload = {
                "controller_name": result.controller_name,
                "scenario_name": result.scenario_name,
                "duration_s": result.duration_s,
                "path_length_m": result.path_length_m,
                "mean_speed_m_s": result.mean_speed_m_s,
                "min_clearance_m": result.min_clearance_m,
                "has_collided": result.has_collided,
                "reached_goal": result.reached_goal,
                "saccade_count": result.saccade_count,
                "mean_computation_time_ms": result.mean_computation_time_ms,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress console logging spam


def start_dashboard(port: int = 8090) -> http.server.HTTPServer:
    """Start web dashboard server."""
    server = socketserver.TCPServer(("", port), DashboardHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server
