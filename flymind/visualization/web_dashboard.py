"""
FlyMind Web Research Dashboard with Interactive WebGL 3D Flight Plane.
Provides an interactive 3D browser console for real-time simulation,
neural activity inspection, 3D trajectory visualization, and benchmark execution.
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
  <title>FlyMind: Insect-Inspired Flight Intelligence 3D Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <style>
    :root {
      --bg: #07090e;
      --card-bg: rgba(18, 22, 32, 0.85);
      --border: rgba(255, 255, 255, 0.12);
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
      padding: 1.1rem 2rem;
      border-bottom: 1px solid var(--border);
      background: rgba(13, 17, 23, 0.90);
      backdrop-filter: blur(14px);
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
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      background: rgba(88, 166, 255, 0.15);
      color: var(--accent);
      border: 1px solid rgba(88, 166, 255, 0.3);
      font-family: 'JetBrains Mono', monospace;
    }
    main {
      padding: 1.75rem 2rem;
      display: grid;
      grid-template-columns: 1fr 390px;
      gap: 1.75rem;
      max-width: 1560px;
      width: 100%;
      margin: 0 auto;
      flex: 1;
    }
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      backdrop-filter: blur(16px);
      padding: 1.5rem;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
      display: flex;
      flex-direction: column;
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
    .viewport-3d-container {
      position: relative;
      background: #030508;
      border-radius: 10px;
      overflow: hidden;
      aspect-ratio: 16/10;
      width: 100%;
      border: 1px solid var(--border);
    }
    #three-canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .viewport-overlay-controls {
      position: absolute;
      bottom: 16px;
      left: 50%;
      transform: translateX(-50%);
      display: flex;
      gap: 8px;
      background: rgba(13, 17, 23, 0.85);
      backdrop-filter: blur(8px);
      padding: 6px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      z-index: 5;
    }
    .view-btn {
      background: rgba(30, 36, 46, 0.85);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.4rem 0.75rem;
      border-radius: 6px;
      font-size: 0.78rem;
      cursor: pointer;
      font-family: inherit;
      transition: all 0.2s ease;
    }
    .view-btn:hover {
      border-color: var(--accent);
      color: var(--accent);
    }
    .view-btn.active {
      background: #1f6feb;
      border-color: var(--accent);
      color: #fff;
    }
    .controls-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
      margin-top: 1.25rem;
    }
    select, button {
      background: rgba(26, 31, 42, 0.85);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.7rem 0.9rem;
      border-radius: 8px;
      font-family: inherit;
      font-size: 0.85rem;
      outline: none;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    select:hover, button:hover {
      border-color: var(--accent);
      background: rgba(36, 44, 60, 0.95);
    }
    button.primary {
      background: linear-gradient(135deg, #1f6feb, #238636);
      border: none;
      font-weight: 600;
      grid-column: span 2;
    }
    button.primary:hover {
      opacity: 0.92;
      box-shadow: 0 4px 14px rgba(35, 134, 54, 0.40);
    }
    .telemetry-row {
      display: flex;
      justify-content: space-between;
      padding: 0.55rem 0;
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
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span class="status-badge"></span>
      <h1>FlyMind</h1>
      <span class="badge">3D Flight Intelligence Platform</span>
    </div>
    <div style="font-size: 0.85rem; color: var(--text-muted);">
      6-DoF Dynamics &bull; 3D Euclidean Space &bull; LPLC2 &bull; Giant Fiber LIF &bull; WTA Gap Selector
    </div>
  </header>

  <main>
    <section class="panel">
      <h2>Interactive 3D Flight Plane & Arena</h2>
      <div class="viewport-3d-container">
        <div id="three-canvas"></div>
        <div class="viewport-overlay-controls">
          <button class="view-btn active" id="btnOrbit" onclick="setCamMode('orbit')">3D Orbit</button>
          <button class="view-btn" id="btnTop" onclick="setCamMode('top')">Top View (XY)</button>
          <button class="view-btn" id="btnSide" onclick="setCamMode('side')">Side (XZ)</button>
          <button class="view-btn" id="btnChase" onclick="setCamMode('chase')">Follow Drone</button>
          <button class="view-btn" id="btnPlay" onclick="togglePlay()">Pause</button>
        </div>
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
        <button class="primary" id="runBtn" onclick="runFlight()">Run Flight Episode in 3D Plane</button>
      </div>
    </section>

    <aside class="panel" style="gap: 1.25rem;">
      <h2>Neural & 3D Telemetry</h2>
      <div class="telemetry-box">
        <div class="telemetry-row">
          <span class="telemetry-label">Flight Mode</span>
          <span class="metric-val" id="modeVal">CRUISE</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Drone 3D Pos (X, Y, Z)</span>
          <span class="metric-val" id="posVal">0.0, 0.0, 1.5 m</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Giant Fiber Vm</span>
          <span class="metric-val" id="gfVal">-65.0 mV</span>
        </div>
        <div class="bar-container">
          <div class="bar-fill" id="gfBar" style="width: 10%;"></div>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Obstacle Clearance</span>
          <span class="metric-val" id="clearVal">4.20 m</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Forward Speed</span>
          <span class="metric-val" id="speedVal">1.50 m/s</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Safety Guard</span>
          <span class="metric-val" style="color: var(--accent-green);" id="safetyVal">NOMINAL</span>
        </div>
      </div>

      <div style="margin-top: auto;">
        <h2>Episode Summary</h2>
        <div id="resultStatus" style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5;">
          Click "Run Flight Episode in 3D Plane" to execute simulated 6-DoF navigation through 3D obstacle arenas.
        </div>
      </div>
    </aside>
  </main>

  <script>
    // -------------------------------------------------------------------------
    // Three.js 3D Plane & Flight Scene Setup
    // -------------------------------------------------------------------------
    const container = document.getElementById('three-canvas');
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 500;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070c);
    scene.fog = new THREE.FogExp2(0x05070c, 0.02);

    const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 100);
    camera.position.set(-6, 8, 12);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(8, 1.5, 0);

    // Lighting
    const ambLight = new THREE.AmbientLight(0xffffff, 0.55);
    scene.add(ambLight);
    const dirLight = new THREE.DirectionalLight(0x58a6ff, 1.1);
    dirLight.position.set(15, 25, 10);
    dirLight.castShadow = true;
    scene.add(dirLight);

    // 3D Ground Plane with Grid
    const grid = new THREE.GridHelper(50, 50, 0x58a6ff, 0x1f242c);
    grid.position.set(10, 0, 0);
    scene.add(grid);

    const groundGeo = new THREE.PlaneGeometry(60, 60);
    const groundMat = new THREE.MeshStandardMaterial({ color: 0x090c12, roughness: 0.9 });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(10, -0.01, 0);
    ground.receiveShadow = true;
    scene.add(ground);

    // Goal Plane
    const goalGeo = new THREE.PlaneGeometry(10, 5);
    const goalMat = new THREE.MeshBasicMaterial({ color: 0x3fb950, transparent: true, opacity: 0.25, side: THREE.DoubleSide });
    const goalMesh = new THREE.Mesh(goalGeo, goalMat);
    goalMesh.position.set(22, 2.0, 0);
    goalMesh.rotation.y = Math.PI / 2;
    scene.add(goalMesh);

    // 3D Quadrotor Drone Model
    const droneGroup = new THREE.Group();
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x161b22, roughness: 0.3 });
    const bodyMesh = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.08, 0.35), bodyMat);
    droneGroup.add(bodyMesh);

    const armMat = new THREE.MeshStandardMaterial({ color: 0x58a6ff });
    droneGroup.add(new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.03, 0.03), armMat));
    droneGroup.add(new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.03, 0.7), armMat));

    const rotorMat = new THREE.MeshBasicMaterial({ color: 0x3fb950, transparent: true, opacity: 0.6 });
    const rotorDiscs = [];
    [[0.32, 0.05, 0], [-0.32, 0.05, 0], [0, 0.05, 0.32], [0, 0.05, -0.32]].forEach(off => {
      const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.01, 16), rotorMat);
      disc.position.set(off[0], off[1], off[2]);
      droneGroup.add(disc);
      rotorDiscs.push(disc);
    });

    const noseArrow = new THREE.Mesh(new THREE.ConeGeometry(0.07, 0.2, 8), new THREE.MeshBasicMaterial({ color: 0xbc8cff }));
    noseArrow.rotation.z = -Math.PI / 2;
    noseArrow.position.set(0.28, 0.02, 0);
    droneGroup.add(noseArrow);

    droneGroup.position.set(0, 1.5, 0);
    scene.add(droneGroup);

    // Dynamic Objects Storage
    let obstacleMeshes = [];
    let trajectoryTube = null;
    let trajectoryPoints = [];
    let frameIdx = 0;
    let isPlaying = true;
    let camMode = 'orbit';

    function clearSceneArena() {
      obstacleMeshes.forEach(m => scene.remove(m));
      obstacleMeshes = [];
      if (trajectoryTube) {
        scene.remove(trajectoryTube);
        trajectoryTube = null;
      }
    }

    function render3DObstacles(obstacles) {
      clearSceneArena();
      obstacles.forEach(obs => {
        const c = obs.center || [0, 0];
        const x = parseFloat(c[0]);
        const z = parseFloat(c[1]);
        const r = parseFloat(obs.radius || 0.35);

        if (obs.type === 'cylinder') {
          const height = 4.0;
          const geo = new THREE.CylinderGeometry(r, r, height, 20);
          const mat = new THREE.MeshStandardMaterial({ color: 0x242b35, roughness: 0.7 });
          const mesh = new THREE.Mesh(geo, mat);
          mesh.position.set(x, height / 2, z);
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          scene.add(mesh);
          obstacleMeshes.push(mesh);

          const ringGeo = new THREE.RingGeometry(r, r + 0.06, 24);
          const ringMat = new THREE.MeshBasicMaterial({ color: 0xf85149, side: THREE.DoubleSide });
          const ring = new THREE.Mesh(ringGeo, ringMat);
          ring.rotation.x = Math.PI / 2;
          ring.position.set(x, 0.02, z);
          scene.add(ring);
          obstacleMeshes.push(ring);
        } else if (obs.type === 'sphere') {
          const y = parseFloat(c[2] || 1.5);
          const geo = new THREE.SphereGeometry(r, 24, 24);
          const mat = new THREE.MeshStandardMaterial({ color: 0xf85149, roughness: 0.3 });
          const mesh = new THREE.Mesh(geo, mat);
          mesh.position.set(x, y, z);
          scene.add(mesh);
          obstacleMeshes.push(mesh);
        }
      });
    }

    function render3DTrajectory(trajectory, hasCollided) {
      if (!trajectory || trajectory.length < 2) return;
      trajectoryPoints = trajectory.map(p => new THREE.Vector3(p[0], p[2], p[1]));

      const curve = new THREE.CatmullRomCurve3(trajectoryPoints);
      const tubeGeo = new THREE.TubeGeometry(curve, trajectoryPoints.length * 2, 0.035, 8, false);
      const tubeMat = new THREE.MeshBasicMaterial({ color: hasCollided ? 0xf85149 : 0x3fb950 });
      trajectoryTube = new THREE.Mesh(tubeGeo, tubeMat);
      scene.add(trajectoryTube);
    }

    // Animation Loop
    function animate() {
      requestAnimationFrame(animate);

      rotorDiscs.forEach(r => r.rotation.y += 0.45);

      if (isPlaying && trajectoryPoints.length > 0) {
        const pt = trajectoryPoints[frameIdx];
        droneGroup.position.copy(pt);

        // Update telemetry
        document.getElementById('posVal').innerText = `${pt.x.toFixed(2)}, ${pt.z.toFixed(2)}, ${pt.y.toFixed(2)} m`;

        frameIdx = (frameIdx + 1) % trajectoryPoints.length;

        if (camMode === 'chase') {
          const chaseOffset = new THREE.Vector3(-3.2, 1.6, 0);
          camera.position.copy(droneGroup.position).add(chaseOffset);
          camera.lookAt(droneGroup.position);
        }
      }

      if (camMode === 'orbit') {
        controls.update();
      }

      renderer.render(scene, camera);
    }
    animate();

    function setCamMode(mode) {
      camMode = mode;
      document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
      if (mode === 'orbit') {
        document.getElementById('btnOrbit').classList.add('active');
        camera.position.set(-6, 8, 12);
        controls.target.set(8, 1.5, 0);
      } else if (mode === 'top') {
        document.getElementById('btnTop').classList.add('active');
        camera.position.set(11, 26, 0);
        controls.target.set(11, 0, 0);
      } else if (mode === 'side') {
        document.getElementById('btnSide').classList.add('active');
        camera.position.set(11, 2.5, 18);
        controls.target.set(11, 1.5, 0);
      } else if (mode === 'chase') {
        document.getElementById('btnChase').classList.add('active');
      }
    }

    function togglePlay() {
      isPlaying = !isPlaying;
      document.getElementById('btnPlay').innerText = isPlaying ? 'Pause' : 'Play';
      document.getElementById('btnPlay').classList.toggle('active', isPlaying);
    }

    async function runFlight() {
      const btn = document.getElementById('runBtn');
      btn.disabled = true;
      btn.innerText = 'Simulating 3D Flight Episode...';
      document.getElementById('resultStatus').innerText = 'Simulating 6-DoF dynamics, bio-perception & 3D space...';

      const ctrl = document.getElementById('ctrlSelect').value;
      const env = document.getElementById('envSelect').value;

      try {
        const resp = await fetch(`/api/run_episode?ctrl=${ctrl}&env=${env}&seed=${Math.floor(Math.random()*1000)}`);
        const data = await resp.json();

        render3DObstacles(data.obstacles || []);
        render3DTrajectory(data.trajectory_3d || [], data.has_collided);
        frameIdx = 0;
        isPlaying = true;

        document.getElementById('modeVal').innerText = data.has_collided ? 'COLLISION' : (data.reached_goal ? 'GOAL_REACHED' : 'HOVER');
        document.getElementById('clearVal').innerText = `${data.min_clearance_m.toFixed(2)} m`;
        document.getElementById('speedVal').innerText = `${data.mean_speed_m_s.toFixed(2)} m/s`;
        document.getElementById('resultStatus').innerHTML = `
          <strong>Result:</strong> ${data.reached_goal ? '<span style="color:#3fb950;">Goal Reached!</span>' : (data.has_collided ? '<span style="color:#f85149;">Collision Detected</span>' : 'Timed Out')}<br>
          <strong>3D Path Length:</strong> ${data.path_length_m.toFixed(1)} m<br>
          <strong>Min Clearance:</strong> ${data.min_clearance_m.toFixed(2)} m<br>
          <strong>Saccade Events:</strong> ${data.saccade_count}<br>
          <strong>Step Latency:</strong> ${data.mean_computation_time_ms.toFixed(2)} ms
        `;
      } catch (err) {
        document.getElementById('resultStatus').innerText = 'Error: ' + err;
      } finally {
        btn.disabled = false;
        btn.innerText = 'Run Flight Episode in 3D Plane';
      }
    }

    window.addEventListener('resize', () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
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

            runner = BenchmarkRunner(dt=0.02, max_episode_time_s=10.0)
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
                "trajectory_3d": result.trajectory_3d,
                "obstacles": result.obstacles,
            }

            def _clean_json(val: Any) -> Any:
                if isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32, np.float16)):
                    return float(val)
                elif isinstance(val, np.ndarray):
                    return [_clean_json(x) for x in val.tolist()]
                elif isinstance(val, (np.bool_, bool)):
                    return bool(val)
                elif isinstance(val, dict):
                    return {str(k): _clean_json(v) for k, v in val.items()}
                elif isinstance(val, (list, tuple)):
                    return [_clean_json(x) for x in val]
                return val

            clean_payload = _clean_json(payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(clean_payload).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


def start_dashboard(port: int = 8090) -> http.server.HTTPServer:
    """Start web dashboard server."""
    server = socketserver.TCPServer(("", port), DashboardHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server
