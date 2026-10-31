"""
FlyMind 3D Trajectory & Arena Visualizer.
Renders 3D flight trajectories, obstacle geometries, and drone attitudes in 3D Euclidean space.
Generates both interactive Three.js 3D HTML models and static publication-grade Matplotlib 3D projections.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from ..evaluation.metrics.performance_metrics import FlightEpisodeMetrics


class Visualizer3D:
    """
    Renders 3D flight paths through obstacle fields.
    """
    def __init__(self, metrics: FlightEpisodeMetrics):
        self.metrics = metrics
        self.trajectory = np.array(metrics.trajectory_3d) if metrics.trajectory_3d else np.zeros((0, 4))
        self.obstacles = metrics.obstacles

    def export_interactive_html(self, output_path: str = "flymind_flight_3d.html") -> str:
        """
        Export a self-contained WebGL 3D interactive plane visualization using Three.js.
        Allows full 360-degree rotation, zooming, panning, and flight playback in any browser.
        """
        traj_data = []
        if len(self.trajectory) > 0:
            for pt in self.trajectory:
                # [x, y, z, yaw] -> Three.js (x, y, z where Y is up)
                # FlyMind world frame: X fwd, Y left, Z up
                # Three.js: X fwd/right, Y up (FlyMind Z), Z side (FlyMind Y)
                traj_data.append([float(pt[0]), float(pt[2]), float(pt[1]), float(pt[3])])

        obs_data = []
        for obs in self.obstacles:
            c = obs.get("center", [0, 0])
            obs_type = obs.get("type", "cylinder")
            obs_data.append({
                "type": obs_type,
                "x": float(c[0]),
                "z": float(c[1]) if len(c) > 1 else 0.0,
                "y": float(c[2]) if len(c) > 2 else 2.0,
                "radius": float(obs.get("radius", 0.35)),
                "height": float(obs.get("z_max", 4.0) - obs.get("z_min", 0.0)),
            })

        payload = {
            "controller": self.metrics.controller_name,
            "scenario": self.metrics.scenario_name,
            "has_collided": self.metrics.has_collided,
            "reached_goal": self.metrics.reached_goal,
            "min_clearance": self.metrics.min_clearance_m,
            "mean_speed": self.metrics.mean_speed_m_s,
            "path_length": self.metrics.path_length_m,
            "duration": self.metrics.duration_s,
            "trajectory": traj_data,
            "obstacles": obs_data,
        }

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>FlyMind 3D Flight Arena - {self.metrics.controller_name}</title>
  <style>
    body {{ margin: 0; overflow: hidden; background: #05070a; font-family: -apple-system, sans-serif; color: #fff; }}
    #hud {{
      position: absolute; top: 16px; left: 16px; z-index: 10;
      background: rgba(15, 20, 28, 0.85); padding: 16px 20px; border-radius: 10px;
      border: 1px solid rgba(255, 255, 255, 0.12); backdrop-filter: blur(8px);
      box-shadow: 0 4px 20px rgba(0,0,0,0.5); max-width: 320px;
    }}
    #hud h1 {{ margin: 0 0 6px 0; font-size: 16px; color: #58a6ff; }}
    #hud p {{ margin: 4px 0; font-size: 13px; color: #8b949e; font-family: monospace; }}
    #hud strong {{ color: #f0f6fc; }}
    #controls {{
      position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 10;
      display: flex; gap: 10px; background: rgba(15, 20, 28, 0.85); padding: 10px 16px;
      border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.12);
    }}
    button {{
      background: #21262d; border: 1px solid #30363d; color: #c9d1d9;
      padding: 6px 14px; border-radius: 6px; font-size: 13px; cursor: pointer;
    }}
    button:hover {{ background: #30363d; color: #58a6ff; }}
    button.active {{ background: #1f6feb; border-color: #58a6ff; color: #fff; }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="hud">
    <h1>FlyMind 3D Flight Plane</h1>
    <p>Controller: <strong>{self.metrics.controller_name}</strong></p>
    <p>Scenario: <strong>{self.metrics.scenario_name.upper()}</strong></p>
    <p>Outcome: <strong style="color: {'#3fb950' if self.metrics.reached_goal else ('#f85149' if self.metrics.has_collided else '#d29922')};">{'GOAL REACHED' if self.metrics.reached_goal else ('COLLISION DETECTED' if self.metrics.has_collided else 'TIMED OUT')}</strong></p>
    <p>Min Clearance: <strong>{self.metrics.min_clearance_m:.2f} m</strong></p>
    <p>Mean Speed: <strong>{self.metrics.mean_speed_m_s:.2f} m/s</strong></p>
    <p>Path Length: <strong>{self.metrics.path_length_m:.2f} m</strong></p>
    <p style="font-size: 11px; margin-top: 8px; color: #58a6ff;">[Left Drag: Orbit] [Right Drag: Pan] [Scroll: Zoom]</p>
  </div>

  <div id="controls">
    <button id="playBtn" class="active" onclick="togglePlay()">Pause</button>
    <button onclick="resetFlight()">Replay</button>
    <button onclick="setCamera('orbit')">3D Orbit</button>
    <button onclick="setCamera('top')">Top View</button>
    <button onclick="setCamera('chase')">Chase Cam</button>
  </div>

  <script>
    const data = {json.dumps(payload)};
    
    // Setup Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x06080d);
    scene.fog = new THREE.FogExp2(0x06080d, 0.02);

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(-8, 9, 14);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    document.body.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(8, 1.5, 0);

    // Lighting
    const ambLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambLight);
    const dirLight = new THREE.DirectionalLight(0x58a6ff, 1.0);
    dirLight.position.set(15, 25, 10);
    dirLight.castShadow = true;
    scene.add(dirLight);

    // Ground Plane with Grid
    const grid = new THREE.GridHelper(50, 50, 0x58a6ff, 0x1f242c);
    grid.position.set(10, 0, 0);
    scene.add(grid);

    const groundGeo = new THREE.PlaneGeometry(60, 60);
    const groundMat = new THREE.MeshStandardMaterial({{ color: 0x080a0f, roughness: 0.9 }});
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    ground.position.set(10, -0.01, 0);
    scene.add(ground);

    // Goal Plane
    const goalGeo = new THREE.PlaneGeometry(12, 6);
    const goalMat = new THREE.MeshBasicMaterial({{ color: 0x3fb950, transparent: true, opacity: 0.25, side: THREE.DoubleSide }});
    const goalMesh = new THREE.Mesh(goalGeo, goalMat);
    goalMesh.position.set(22, 2.5, 0);
    goalMesh.rotation.y = Math.PI / 2;
    scene.add(goalMesh);

    // Render 3D Obstacles
    data.obstacles.forEach(obs => {{
      if (obs.type === 'cylinder') {{
        const geo = new THREE.CylinderGeometry(obs.radius, obs.radius, obs.height, 24);
        const mat = new THREE.MeshStandardMaterial({{
          color: 0x2e3440,
          roughness: 0.6,
          metalness: 0.2
        }});
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(obs.x, obs.height / 2, obs.z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);

        // Glowing boundary ring
        const ringGeo = new THREE.RingGeometry(obs.radius, obs.radius + 0.05, 32);
        const ringMat = new THREE.MeshBasicMaterial({{ color: 0xf85149, side: THREE.DoubleSide }});
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = Math.PI / 2;
        ring.position.set(obs.x, 0.02, obs.z);
        scene.add(ring);
      }} else if (obs.type === 'sphere') {{
        const geo = new THREE.SphereGeometry(obs.radius, 24, 24);
        const mat = new THREE.MeshStandardMaterial({{ color: 0xf85149, roughness: 0.3 }});
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(obs.x, obs.y, obs.z);
        scene.add(mesh);
      }}
    }});

    // Draw 3D Trajectory Path
    if (data.trajectory.length > 1) {{
      const points = data.trajectory.map(p => new THREE.Vector3(p[0], p[1], p[2]));
      const curve = new THREE.CatmullRomCurve3(points);
      const tubeGeo = new THREE.TubeGeometry(curve, points.length * 2, 0.04, 8, false);
      const tubeMat = new THREE.MeshBasicMaterial({{ color: data.has_collided ? 0xf85149 : 0x3fb950 }});
      const tube = new THREE.Mesh(tubeGeo, tubeMat);
      scene.add(tube);
    }}

    // 3D Quadrotor Drone Model
    const droneGroup = new THREE.Group();
    // Central Fuselage
    const bodyGeo = new THREE.BoxGeometry(0.35, 0.08, 0.35);
    const bodyMat = new THREE.MeshStandardMaterial({{ color: 0x161b22, roughness: 0.4 }});
    const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
    droneGroup.add(bodyMesh);

    // Cross Arms
    const armMat = new THREE.MeshStandardMaterial({{ color: 0x58a6ff }});
    const arm1 = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.03, 0.03), armMat);
    const arm2 = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.03, 0.7), armMat);
    droneGroup.add(arm1);
    droneGroup.add(arm2);

    // 4 Rotor Discs
    const rotorMat = new THREE.MeshBasicMaterial({{ color: 0x3fb950, transparent: true, opacity: 0.6 }});
    const rotorDiscs = [];
    const offsets = [[0.32, 0.05, 0], [-0.32, 0.05, 0], [0, 0.05, 0.32], [0, 0.05, -0.32]];
    offsets.forEach(off => {{
      const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.01, 16), rotorMat);
      disc.position.set(off[0], off[1], off[2]);
      droneGroup.add(disc);
      rotorDiscs.push(disc);
    }});

    // Forward Direction Arrow (Nose)
    const arrowGeo = new THREE.ConeGeometry(0.08, 0.2, 8);
    const arrowMat = new THREE.MeshBasicMaterial({{ color: 0xbc8cff }});
    const arrow = new THREE.Mesh(arrowGeo, arrowMat);
    arrow.rotation.z = -Math.PI / 2;
    arrow.position.set(0.28, 0.02, 0);
    droneGroup.add(arrow);

    scene.add(droneGroup);

    // Flight Animation
    let frameIdx = 0;
    let isPlaying = true;
    let camMode = 'orbit';

    function animate() {{
      requestAnimationFrame(animate);

      // Spin rotors
      rotorDiscs.forEach(r => r.rotation.y += 0.4);

      if (isPlaying && data.trajectory.length > 0) {{
        const pt = data.trajectory[frameIdx];
        droneGroup.position.set(pt[0], pt[1], pt[2]);
        droneGroup.rotation.y = -pt[3]; // yaw

        frameIdx = (frameIdx + 1) % data.trajectory.length;

        if (camMode === 'chase') {{
          const chaseOffset = new THREE.Vector3(-3.5, 1.8, 0);
          chaseOffset.applyAxisAngle(new THREE.Vector3(0, 1, 0), -pt[3]);
          camera.position.copy(droneGroup.position).add(chaseOffset);
          camera.lookAt(droneGroup.position);
        }}
      }}

      if (camMode === 'orbit') {{
        controls.update();
      }}

      renderer.render(scene, camera);
    }}
    animate();

    function togglePlay() {{
      isPlaying = !isPlaying;
      document.getElementById('playBtn').innerText = isPlaying ? 'Pause' : 'Play';
      document.getElementById('playBtn').classList.toggle('active', isPlaying);
    }}

    function resetFlight() {{
      frameIdx = 0;
      isPlaying = true;
      document.getElementById('playBtn').innerText = 'Pause';
      document.getElementById('playBtn').classList.add('active');
    }}

    function setCamera(mode) {{
      camMode = mode;
      if (mode === 'top') {{
        camera.position.set(10, 26, 0);
        controls.target.set(10, 0, 0);
      }} else if (mode === 'orbit') {{
        camera.position.set(-6, 8, 12);
        controls.target.set(8, 1.5, 0);
      }}
    }}

    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});
  </script>
</body>
</html>
"""
        out = Path(output_path)
        out.write_text(html, encoding="utf-8")
        return str(out.resolve())
