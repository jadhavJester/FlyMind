"""
FlyMind Cinematic Simulation Video Recorder.
Renders closed-loop 6-DoF quadrotor flight with biological perception overlays,
chase camera 3D tracking, nearness heatmaps, and telemetry HUD into 720p HD MP4 and WebP.
"""
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
import imageio

# Add root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flymind.simulation.quadrotor_6dof import Quadrotor6DoF
from flymind.simulation.sensors.synthetic_camera import SyntheticCamera
from flymind.simulation.sensors.noisy_imu import NoisyIMU
from flymind.simulation.environments.obstacle_forest import ObstacleForest
from flymind.simulation.environments.dynamic_looming import DynamicLoomingArena
from flymind.simulation.environments.narrow_gates import NarrowGatesCorridor

from flymind.perception.optical_flow.farneback_dense import FarnebackDenseFlow
from flymind.perception.elementary_motion_detection.hr_emd import HRElementaryMotionDetector
from flymind.perception.looming_detection.angular_expansion import LoomingDetector
from flymind.perception.looming_detection.imu_derotation import IMUFlowDerotator
from flymind.perception.nearness_estimation.flow_divergence import FlowDivergenceNearness

from flymind.bioinspired.lplc2.population_coding import LPLC2Population
from flymind.bioinspired.neural_models.giant_fiber_lif import GiantFiberNeuron
from flymind.bioinspired.motion_detectors.wta_gap import WinnerTakeAllGapSelector

from flymind.navigation.obstacle_awareness.collision_risk import CollisionRiskAssessor
from flymind.navigation.safe_navigation.reactive_navigator import ReactiveNavigator
from flymind.safety.safety_supervisor.supervisor import SafetySupervisor


def draw_drone_in_chase_cam(chase_img, cam: SyntheticCamera, drone_pos: np.ndarray, drone_euler: np.ndarray):
    """Projects a 3D quadrotor body frame with rotors and LED markers onto the chase camera."""
    roll, pitch, yaw = drone_euler
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R_bw = np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp, sr * cp, cr * cp]
    ])

    arm_len = 0.35
    body_points = [
        np.array([arm_len, 0.0, 0.0]),   # Front (+X)
        np.array([-arm_len, 0.0, 0.0]),  # Back (-X)
        np.array([0.0, arm_len, 0.0]),   # Left (+Y)
        np.array([0.0, -arm_len, 0.0]),  # Right (-Y)
        np.array([0.0, 0.0, 0.08]),      # Center mast (+Z)
    ]

    chase_euler = np.array([0.0, 0.18, yaw])
    chase_pos = drone_pos - 2.8 * np.array([np.cos(yaw), np.sin(yaw), 0.0]) + np.array([0.0, 0.0, 1.1])

    proj_pts = []
    for pt in body_points:
        world_pt = drone_pos + R_bw @ pt
        cam_pt = cam.world_to_camera(world_pt, chase_pos, chase_euler)
        if cam_pt[2] > 0.1:
            u = int(cam.cx + (cam_pt[0] / cam_pt[2]) * cam.fx)
            v = int(cam.cy + (cam_pt[1] / cam_pt[2]) * cam.fy)
            proj_pts.append((u, v))
        else:
            proj_pts.append(None)

    c_pt = proj_pts[4]
    f_pt, b_pt, l_pt, r_pt = proj_pts[0], proj_pts[1], proj_pts[2], proj_pts[3]

    if c_pt and f_pt and b_pt and l_pt and r_pt:
        # Cross arms
        cv2.line(chase_img, f_pt, b_pt, (50, 50, 240), 3)  # Red front-back
        cv2.line(chase_img, l_pt, r_pt, (200, 200, 200), 2)  # Gray left-right
        cv2.circle(chase_img, c_pt, 5, (0, 255, 255), -1)  # Yellow center

        # Rotors
        for rotor_pt, col in [(f_pt, (0, 0, 255)), (b_pt, (0, 255, 0)), (l_pt, (255, 0, 0)), (r_pt, (255, 0, 0))]:
            cv2.circle(chase_img, rotor_pt, 12, col, 2)
            cv2.circle(chase_img, rotor_pt, 3, (255, 255, 255), -1)


def draw_hud(frame: np.ndarray, current_time: float, min_dist: float, speed: float, alt: float,
             mode_str: str, gf_v: float, gf_spiking: bool, hist_gf: list):
    """Renders bottom telemetry panel and neural oscilloscope."""
    h_start = 540
    frame[h_start:, :] = (20, 24, 28)
    cv2.line(frame, (0, h_start), (1280, h_start), (60, 70, 80), 2)

    # 1. Flight Status Badges
    cv2.putText(frame, "FLYMIND AUTONOMOUS BIO-FLIGHT INTELLIGENCE", (25, h_start + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (230, 230, 230), 1)

    # State Badge
    badge_col = (0, 210, 0) if "CRUISE" in mode_str else ((0, 140, 255) if "GAP" in mode_str else (0, 0, 240))
    cv2.rectangle(frame, (25, h_start + 45), (280, h_start + 85), badge_col, -1, cv2.LINE_AA)
    cv2.putText(frame, f"MODE: {mode_str}", (35, h_start + 72),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

    # Clearance gauge
    clear_col = (0, 255, 0) if min_dist > 1.0 else ((0, 200, 255) if min_dist > 0.5 else (0, 0, 255))
    cv2.putText(frame, f"MIN CLEARANCE: {min_dist:.2f} m", (25, h_start + 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, clear_col, 2)
    cv2.putText(frame, f"GROUND SPEED:  {speed:.2f} m/s", (25, h_start + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"ALTITUDE:      {alt:.2f} m", (25, h_start + 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # 2. Giant Fiber LIF Oscilloscope (middle-bottom)
    cv2.putText(frame, "GIANT FIBER LIF MEMBRANE POTENTIAL (Vm)", (350, h_start + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (180, 210, 255), 1)
    osc_rect = (350, h_start + 45, 450, 115)
    cv2.rectangle(frame, (osc_rect[0], osc_rect[1]), (osc_rect[0] + osc_rect[2], osc_rect[1] + osc_rect[3]), (10, 15, 20), -1)
    cv2.rectangle(frame, (osc_rect[0], osc_rect[1]), (osc_rect[0] + osc_rect[2], osc_rect[1] + osc_rect[3]), (50, 60, 70), 1)

    # Threshold line (-45 mV threshold, -70 to -40 mV scale)
    v_norm_th = (-45.0 - (-70.0)) / 30.0
    v_th_y = int(osc_rect[1] + osc_rect[3] - v_norm_th * osc_rect[3])
    cv2.line(frame, (osc_rect[0], v_th_y), (osc_rect[0] + osc_rect[2], v_th_y), (0, 0, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "-45 mV (V_thresh)", (osc_rect[0] + 5, v_th_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)

    if len(hist_gf) > 1:
        pts = []
        for i, val in enumerate(hist_gf[-90:]):
            x = osc_rect[0] + int((i / 90.0) * osc_rect[2])
            norm_v = np.clip((val - (-70.0)) / 30.0, 0.05, 0.95)
            y = int(osc_rect[1] + osc_rect[3] - norm_v * osc_rect[3])
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], (0, 240, 240), 2, cv2.LINE_AA)

    # Spiking Alert
    if gf_spiking:
        cv2.putText(frame, ">>> SACCADE TRIGGERED <<<", (osc_rect[0] + 160, v_th_y - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 255), 2)

    # 3. Bio-Vision Architecture Specs (right-bottom)
    cv2.putText(frame, "BIOMORPHIC CIRCUITS", (850, h_start + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1)
    specs = [
        "Lobula Plate: 4-Channel HR-EMD (218 Hz)",
        "Looming: LPLC2 Cardinal Multiplicative Gating",
        "Escape Trigger: Giant Fiber LIF (tau=15ms)",
        "Steering: 8-Sector WTA Lateral Flow Minimization",
        "Ego-Compensation: 6-Axis IMU Flow Derotation",
        "Safety Interlock: Hardware-Isolated Supervisor"
    ]
    for idx, sp in enumerate(specs):
        cv2.putText(frame, f"- {sp}", (850, h_start + 55 + idx * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 190), 1)


def record_cinematic_flight(scenario_name: str = "forest", max_seconds: float = 12.0) -> Tuple[str, str]:
    print(f"\n=======================================================")
    print(f" RECORDING FLYMIND SIMULATION VIDEO: '{scenario_name.upper()}'")
    print(f"=======================================================\n")

    videos_dir = PROJECT_ROOT / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    dt = 0.02
    seed = 42

    # Instantiate Environment
    if scenario_name == "forest":
        env = ObstacleForest(num_obstacles=25, seed=seed)
        init_pos = np.array([0.0, 0.0, 1.5], dtype=np.float32)
        goal_x = 22.0
    elif scenario_name == "gates":
        env = NarrowGatesCorridor(gate_spacing=7.0, slit_width=1.2, seed=seed)
        init_pos = np.array([0.0, 0.0, 1.5], dtype=np.float32)
        goal_x = 20.0
    elif scenario_name == "looming":
        env = DynamicLoomingArena(approach_speed=2.5, initial_distance=15.0)
        init_pos = np.array([0.0, 0.0, 1.5], dtype=np.float32)
        goal_x = 18.0
    else:
        raise ValueError(f"Unknown environment: {scenario_name}")

    # Quadrotor, Sensors, and Perception
    quad = Quadrotor6DoF(dt=dt, initial_pos=init_pos)
    onboard_cam = SyntheticCamera(width=240, height=180, fov_deg=90.0)
    chase_cam = SyntheticCamera(width=800, height=540, fov_deg=80.0)
    imu = NoisyIMU(dt=dt)

    farneback = FarnebackDenseFlow()
    emd = HRElementaryMotionDetector(dt=dt)
    looming_det = LoomingDetector()
    derotator = IMUFlowDerotator(width=240, height=180, fov_deg=90.0)
    nearness_est = FlowDivergenceNearness()
    lplc2_pop = LPLC2Population(image_width=240, image_height=180)
    gf_neuron = GiantFiberNeuron()
    wta_gap = WinnerTakeAllGapSelector(num_sectors=8)
    risk_assessor = CollisionRiskAssessor(num_sectors=8)
    reactive_nav = ReactiveNavigator(nominal_cruise_speed=1.5)
    safety_sup = SafetySupervisor()

    frames = []
    hist_gf = []
    current_time = 0.0
    total_steps = int(max_seconds / dt)

    print(f"Simulating & rendering {total_steps} flight steps @ 30 FPS...")

    for step in range(total_steps):
        # 1. Environment obstacles & onboard camera
        obstacles = env.get_obstacles(current_time)
        img_onboard, _ = onboard_cam.render(quad.pos, quad.euler, obstacles)
        omega_meas, _ = imu.measure(quad.omega, quad.rot_mat @ np.array([0.0, 0.0, -9.81]))
        min_dist = env.check_clearance(quad.pos)

        # 2. Perception & Bio-vision pipeline
        flow_u, flow_v = farneback.compute(img_onboard)
        u_trans, v_trans = derotator.derotate(flow_u, flow_v, omega_meas, dt=dt)
        fwd_speed = float(np.linalg.norm(quad.vel[:2]))
        nearness_map = nearness_est.compute(u_trans, v_trans, fwd_speed, dt=dt)
        sec_nearness = nearness_est.compute_sector_nearness(nearness_map, num_sectors=8)

        emd_signals = emd.process_frame(img_onboard)
        lplc2_resp = lplc2_pop.process(emd_signals)
        gf_state = gf_neuron.step(lplc2_resp.active_count, lplc2_resp.mean_activation, dt=dt)
        hist_gf.append(gf_state.membrane_potential_mv)

        gap_dec = wta_gap.select_gap(sec_nearness)
        looming_state = looming_det.detect_looming(img_onboard, u_trans, v_trans, current_time)
        risk_map = risk_assessor.evaluate(sec_nearness, looming_state, lplc2_resp)

        setpoint, trans = reactive_nav.compute_setpoints(
            dt=dt,
            gf_spike=gf_state.is_spiking,
            risk_map=risk_map,
            gap_decision=gap_dec,
            threat_centroid_x=looming_state.dominant_threat.center_x if looming_state.dominant_threat else 0.0,
            current_altitude=quad.pos[2],
        )

        safety_sup.heartbeat(current_time)
        intervention = safety_sup.verify_and_enforce(
            setpoint=setpoint,
            current_position=quad.pos,
            estimated_clearance_m=min_dist,
            current_time=current_time,
        )
        enforced = intervention.enforced_setpoint

        quad.step_velocity_body(
            vx_body=enforced.vx_body,
            vy_body=enforced.vy_body,
            vz_body=enforced.vz_body,
            yaw_rate_cmd=enforced.yaw_rate,
        )

        # 3. Compose 1280x720 Cinematic HD Frame
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Main Panel (Top-Left 800x540): 3D Chase Camera
        yaw = quad.yaw
        chase_euler = np.array([0.0, 0.18, yaw])
        chase_pos = quad.pos - 2.8 * np.array([np.cos(yaw), np.sin(yaw), 0.0]) + np.array([0.0, 0.0, 1.1])
        chase_img, _ = chase_cam.render(chase_pos, chase_euler, obstacles)
        draw_drone_in_chase_cam(chase_img, chase_cam, quad.pos, quad.euler)
        cv2.putText(chase_img, "CHASE 3D CAMERA", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 1)
        canvas[:540, :800] = chase_img

        # Panel 2 (Top-Right 480x270): Onboard Camera with Flow Quiver Overlay
        cockpit_res = cv2.resize(img_onboard, (480, 270))
        # Draw optical flow quiver
        step_sz = 24
        for y_f in range(0, 180, step_sz):
            for x_f in range(0, 240, step_sz):
                dx = int(u_trans[y_f, x_f] * 2.5)
                dy = int(v_trans[y_f, x_f] * 2.5)
                if abs(dx) > 1 or abs(dy) > 1:
                    x_c = int(x_f * 2.0)
                    y_c = int(y_f * 1.5)
                    cv2.arrowedLine(cockpit_res, (x_c, y_c), (x_c + dx, y_c + dy), (0, 255, 0), 1, tipLength=0.3)
        # Looming threat box
        if looming_state.dominant_threat and looming_state.is_imminent_danger:
            th = looming_state.dominant_threat
            bx = int((th.center_x + 1.0) * 0.5 * 480)
            by = int((th.center_y + 0.75) * 0.66 * 270)
            cv2.rectangle(cockpit_res, (bx - 40, by - 40), (bx + 40, by + 40), (0, 0, 255), 2)
            cv2.putText(cockpit_res, f"LOOMING TTC: {th.ttc_seconds:.1f}s", (bx - 60, by - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        cv2.putText(cockpit_res, "ONBOARD BIO-CAMERA & FLOW", (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)
        canvas[:270, 800:] = cockpit_res

        # Panel 3 (Middle-Right 480x270): Flow Divergence Nearness Heatmap
        norm_nearness = np.clip(nearness_map / 2.5, 0.0, 1.0)
        nearness_u8 = (norm_nearness * 255).astype(np.uint8)
        color_nearness = cv2.applyColorMap(nearness_u8, cv2.COLORMAP_TURBO)
        color_nearness_res = cv2.resize(color_nearness, (480, 270))
        cv2.putText(color_nearness_res, "FLOW DIVERGENCE (NEARNESS TTC)", (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)
        canvas[270:540, 800:] = color_nearness_res

        # Bottom Panel (1280x180): Telemetry HUD & Oscilloscope
        mode_str = trans.current_mode.name
        draw_hud(
            frame=canvas,
            current_time=current_time,
            min_dist=min_dist,
            speed=fwd_speed,
            alt=quad.pos[2],
            mode_str=mode_str,
            gf_v=gf_state.membrane_potential_mv,
            gf_spiking=gf_state.is_spiking,
            hist_gf=hist_gf,
        )

        # Convert BGR to RGB for video encoding
        rgb_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)

        if step % 50 == 0:
            print(f"  Step {step}/{total_steps} (t = {current_time:.2f}s, min_clear = {min_dist:.2f}m)...", flush=True)

        if quad.pos[0] >= goal_x:
            print(f"Goal reached at step {step}!")
            break
        if min_dist <= 0.18:
            print(f"Contact registered at step {step}!")
            break

        current_time += dt

    # Save output video in MP4 and WebP formats
    mp4_path = videos_dir / f"flymind_{scenario_name}.mp4"
    webp_path = videos_dir / f"flymind_{scenario_name}.webp"

    print(f"\nWriting MP4 video to {mp4_path}...")
    imageio.mimsave(str(mp4_path), frames, fps=30, format="mp4")

    print(f"Writing WebP animation to {webp_path}...")
    # Downsample frames slightly for token-efficient WebP playback
    webp_frames = [cv2.resize(f, (640, 360)) for f in frames[::2]]
    imageio.mimsave(str(webp_path), webp_frames, fps=15, format="webp")

    # Also copy to Artifact directory for IDE inline presentation
    artifact_dir = Path(r"C:\Users\Udit Jadhav\.gemini\antigravity-ide\brain\429769f3-3e3d-45be-b27a-250c0eafa2cd")
    if artifact_dir.exists():
        art_mp4 = artifact_dir / f"flymind_{scenario_name}.mp4"
        art_webp = artifact_dir / f"flymind_{scenario_name}.webp"
        shutil.copyfile(mp4_path, art_mp4)
        shutil.copyfile(webp_path, art_webp)
        print(f"Copied to artifact directory: {art_webp}")

    print(f"\n[DONE] Successfully recorded {len(frames)} frames ({len(frames)/30.0:.1f}s)!")
    return str(mp4_path), str(webp_path)


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "forest"
    record_cinematic_flight(scenario_name=scenario, max_seconds=10.0)
