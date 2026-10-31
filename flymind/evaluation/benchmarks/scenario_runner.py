"""
Automated Multi-Seed Benchmark Runner.
Executes batch simulations across all 5 navigation controllers and 3 challenge environments.
"""
import time
from enum import Enum, auto
from typing import Dict, List, Optional
import numpy as np

from ...simulation.quadrotor_6dof import Quadrotor6DoF
from ...simulation.sensors.synthetic_camera import SyntheticCamera
from ...simulation.sensors.noisy_imu import NoisyIMU
from ...simulation.environments.obstacle_forest import ObstacleForest
from ...simulation.environments.narrow_gates import NarrowGatesCorridor
from ...simulation.environments.dynamic_looming import DynamicLoomingArena

from ...perception.optical_flow.farneback_dense import FarnebackDenseFlow
from ...perception.elementary_motion_detection.hr_emd import HRElementaryMotionDetector
from ...perception.looming_detection.angular_expansion import LoomingDetector
from ...perception.looming_detection.imu_derotation import IMUFlowDerotator
from ...perception.nearness_estimation.flow_divergence import FlowDivergenceNearness

from ...bioinspired.lplc2.population_coding import LPLC2Population
from ...bioinspired.neural_models.giant_fiber_lif import GiantFiberNeuron
from ...bioinspired.motion_detectors.wta_gap import WinnerTakeAllGapSelector

from ...navigation.obstacle_awareness.collision_risk import CollisionRiskAssessor
from ...navigation.safe_navigation.reactive_navigator import ReactiveNavigator
from ...safety.safety_supervisor.supervisor import SafetySupervisor

from ...baselines.opencv_edge_detector import CannyEdgeBaseline
from ...baselines.monocular_depth_net import MonocularDepthBaseline

from ..metrics.performance_metrics import FlightEpisodeMetrics, MetricsCollector


class ControllerType(Enum):
    FLYMIND_FULL = "FlyMindBio (Full)"
    LPLC2_ONLY = "LPLC2-GF Only"
    WTA_FLOW_ONLY = "WTA Flow Gap Only"
    CANNY_BASELINE = "Canny Edge Baseline"
    DEPTHNET_BASELINE = "Monocular DepthNet Baseline"


class BenchmarkRunner:
    """
    Executes controlled, reproducible simulation flights across controllers.
    """
    def __init__(self, dt: float = 0.02, max_episode_time_s: float = 12.0):
        self.dt = dt
        self.max_episode_time_s = max_episode_time_s

    def run_episode(
        self,
        controller_type: ControllerType,
        environment_name: str = "forest",
        seed: int = 42,
    ) -> FlightEpisodeMetrics:
        """
        Run a single flight episode.
        """
        np.random.seed(seed)
        collector = MetricsCollector(
            controller_name=controller_type.value,
            scenario_name=environment_name,
            seed=seed,
        )

        # 1. Instantiate environment
        if environment_name == "forest":
            env = ObstacleForest(num_obstacles=25, seed=seed)
            init_pos = np.array([0.0, 0.0, 1.5], dtype=np.float32)
            goal_x = 22.0
        elif environment_name == "gates":
            env = NarrowGatesCorridor(gate_spacing=7.0, slit_width=1.2, seed=seed)
            init_pos = np.array([0.0, 0.0, 1.5], dtype=np.float32)
            goal_x = 20.0
        elif environment_name == "looming":
            env = DynamicLoomingArena(approach_speed=2.5, initial_distance=15.0)
            init_pos = np.array([0.0, 0.0, 1.5], dtype=np.float32)
            goal_x = 18.0
        else:
            raise ValueError(f"Unknown environment: {environment_name}")

        # 2. Instantiate simulation vehicle and sensors
        quad = Quadrotor6DoF(dt=self.dt, initial_pos=init_pos)
        cam = SyntheticCamera(width=160, height=120, fov_deg=90.0)
        imu = NoisyIMU(dt=self.dt)

        # 3. Instantiate Perception & Bio modules
        farneback = FarnebackDenseFlow()
        emd = HRElementaryMotionDetector(dt=self.dt)
        looming_det = LoomingDetector()
        derotator = IMUFlowDerotator(width=160, height=120, fov_deg=90.0)
        nearness_est = FlowDivergenceNearness()
        lplc2_pop = LPLC2Population(image_width=160, image_height=120)
        gf_neuron = GiantFiberNeuron()
        wta_gap = WinnerTakeAllGapSelector(num_sectors=8)
        risk_assessor = CollisionRiskAssessor(num_sectors=8)
        reactive_nav = ReactiveNavigator(nominal_cruise_speed=1.5)
        safety_sup = SafetySupervisor()

        # Baselines
        canny_base = CannyEdgeBaseline(nominal_speed=1.5)
        depth_base = MonocularDepthBaseline(nominal_speed=1.5)

        current_time = 0.0
        steps = int(self.max_episode_time_s / self.dt)

        for step in range(steps):
            t_start = time.perf_counter()

            # Render camera frame from environment
            obstacles = env.get_obstacles(current_time)
            img, depth = cam.render(quad.pos, quad.yaw, obstacles)

            # Get IMU measurement
            omega_meas, accel_meas = imu.measure(quad.omega, quad.rot_mat @ np.array([0.0, 0.0, -9.81]))

            # Check min physical clearance to obstacles
            min_dist = env.check_clearance(quad.pos)
            if min_dist <= 0.20:
                collector.register_collision()
                break

            if quad.pos[0] >= goal_x:
                collector.register_goal_reached()
                break

            # Controller computation
            is_saccade = False

            if controller_type == ControllerType.FLYMIND_FULL:
                # 1. Optical flow
                flow_u, flow_v = farneback.compute(img)
                # 2. IMU derotation
                u_trans, v_trans = derotator.derotate(flow_u, flow_v, omega_meas, dt=self.dt)
                # 3. Nearness map
                fwd_speed = float(quad.vel[0])
                nearness_map = nearness_est.compute(u_trans, v_trans, fwd_speed, dt=self.dt)
                sec_nearness = nearness_est.compute_sector_nearness(nearness_map, num_sectors=8)
                # 4. HR-EMD
                emd_signals = emd.process_frame(img)
                # 5. LPLC2 population
                lplc2_resp = lplc2_pop.process(emd_signals)
                # 6. Giant Fiber LIF
                gf_state = gf_neuron.step(lplc2_resp.active_count, lplc2_resp.mean_activation, dt=self.dt)
                if gf_state.is_spiking:
                    collector.register_saccade()
                # 7. WTA gap selection
                gap_dec = wta_gap.select_gap(sec_nearness)
                # 8. Looming detector
                looming_state = looming_det.detect_looming(img, u_trans, v_trans, current_time)
                # 9. Spatial risk
                risk_map = risk_assessor.evaluate(sec_nearness, looming_state, lplc2_resp)
                # 10. Reactive setpoints
                setpoint, trans = reactive_nav.compute_setpoints(
                    dt=self.dt,
                    gf_spike=gf_state.is_spiking,
                    risk_map=risk_map,
                    gap_decision=gap_dec,
                    threat_centroid_x=looming_state.dominant_threat.center_x if looming_state.dominant_threat else 0.0,
                    current_altitude=quad.pos[2],
                )
                is_saccade = (trans.current_mode.name == "EVASIVE_SACCADE")

            elif controller_type == ControllerType.LPLC2_ONLY:
                emd_signals = emd.process_frame(img)
                lplc2_resp = lplc2_pop.process(emd_signals)
                gf_state = gf_neuron.step(lplc2_resp.active_count, lplc2_resp.mean_activation, dt=self.dt)
                if gf_state.is_spiking:
                    collector.register_saccade()
                    is_saccade = True
                flow_u, flow_v = farneback.compute(img)
                looming_state = looming_det.detect_looming(img, flow_u, flow_v, current_time)
                # Null gap selector (drive straight)
                sec_near = np.zeros(8, dtype=np.float32)
                gap_dec = wta_gap.select_gap(sec_near)
                risk_map = risk_assessor.evaluate(sec_near, looming_state, lplc2_resp)
                setpoint, _ = reactive_nav.compute_setpoints(
                    dt=self.dt,
                    gf_spike=gf_state.is_spiking,
                    risk_map=risk_map,
                    gap_decision=gap_dec,
                    threat_centroid_x=looming_state.dominant_threat.center_x if looming_state.dominant_threat else 0.0,
                    current_altitude=quad.pos[2],
                )

            elif controller_type == ControllerType.WTA_FLOW_ONLY:
                flow_u, flow_v = farneback.compute(img)
                nearness_map = nearness_est.compute(flow_u, flow_v, float(quad.vel[0]), dt=self.dt)
                sec_near = nearness_est.compute_sector_nearness(nearness_map, num_sectors=8)
                gap_dec = wta_gap.select_gap(sec_near)
                # No looming detection or GF escape
                looming_state = looming_det.detect_looming(img, flow_u, flow_v, current_time)
                dummy_lplc2 = lplc2_pop.process({"right": np.zeros((120, 160)), "left": np.zeros((120, 160)), "up": np.zeros((120, 160)), "down": np.zeros((120, 160))})
                risk_map = risk_assessor.evaluate(sec_near, looming_state, dummy_lplc2)
                setpoint, _ = reactive_nav.compute_setpoints(
                    dt=self.dt,
                    gf_spike=False,
                    risk_map=risk_map,
                    gap_decision=gap_dec,
                    threat_centroid_x=0.0,
                    current_altitude=quad.pos[2],
                )

            elif controller_type == ControllerType.CANNY_BASELINE:
                setpoint = canny_base.compute_setpoint(img, quad.pos[2])

            elif controller_type == ControllerType.DEPTHNET_BASELINE:
                setpoint = depth_base.compute_setpoint(img, quad.pos[2])

            else:
                raise ValueError("Unknown controller type")

            # Safety supervisor enforcement
            safety_sup.heartbeat(current_time)
            intervention = safety_sup.verify_and_enforce(
                setpoint=setpoint,
                current_position=quad.pos,
                estimated_clearance_m=min_dist,
                current_time=current_time,
            )
            enforced = intervention.enforced_setpoint

            # Step 6-DoF vehicle dynamics
            quad.step_velocity_body(
                vx_body=enforced.vx_body,
                vy_body=enforced.vy_body,
                vz_body=enforced.vz_body,
                yaw_rate_cmd=enforced.yaw_rate,
            )

            t_end = time.perf_counter()
            step_duration_ms = (t_end - t_start) * 1000.0

            collector.record_step(
                position=quad.pos,
                velocity=quad.vel,
                clearance_m=min_dist,
                step_time_ms=step_duration_ms,
                is_saccading=is_saccade,
                current_time_s=current_time,
            )

            current_time += self.dt

        return collector.compute_summary()
