"""
Synthetic Forward-Facing Monocular Camera with Depth Buffer.

Implements pinhole projection rendering 3D environmental obstacles (cylinders, spheres,
corridors, and gates) into 2D RGB frames and ground-truth depth maps.
Supports lighting modulation: day, overcast, dim, and dark/night.
"""

from __future__ import annotations
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple


class SyntheticCamera:
    """
    Simulated Monocular Forward Camera.
    Mounted at body frame: X forward, Y left, Z up.
    Produces RGB frames (H x W x 3) and Depth maps (H x W) in meters.
    """

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        width: int | None = None,
        height: int | None = None,
        fov_deg: float | None = None,
    ):
        cfg = (config or {}).get("camera", {})
        res = cfg.get("resolution", [160, 120])
        self.width = int(width if width is not None else res[0])
        self.height = int(height if height is not None else res[1])
        default_fov = float(fov_deg if fov_deg is not None else cfg.get("fov_horizontal_deg", 90.0))
        self.fov_h = default_fov
        self.fov_v = float(cfg.get("fov_vertical_deg", default_fov * 0.75))
        self.max_dist = float(cfg.get("max_render_distance_m", 15.0))
        self.min_dist = float(cfg.get("min_render_distance_m", 0.15))

        # Camera intrinsic matrix parameters
        self.fx = (self.width / 2.0) / np.tan(np.deg2rad(self.fov_h / 2.0))
        self.fy = (self.height / 2.0) / np.tan(np.deg2rad(self.fov_v / 2.0))
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0


    def world_to_camera(self, point_world: np.ndarray, drone_pos: np.ndarray, drone_euler: np.ndarray) -> np.ndarray:
        """
        Transforms 3D world coordinate to camera frame:
        Camera Frame convention: X right, Y down, Z forward.
        """
        # Translation
        p_rel = point_world - drone_pos

        # Body rotation (ZYX)
        roll, pitch, yaw = drone_euler
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R_body_to_world = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,               cp * cr]
        ], dtype=np.float64)

        p_body = R_body_to_world.T @ p_rel

        # Body (X fwd, Y left, Z up) -> Camera (X right, Y down, Z fwd)
        p_cam = np.array([-p_body[1], -p_body[2], p_body[0]], dtype=np.float64)
        return p_cam

    def render(
        self,
        drone_pos: np.ndarray,
        drone_euler: np.ndarray | float,
        obstacles: List[Dict[str, Any]],
        lighting: str = "day"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Renders the synthetic scene into RGB image and metric depth buffer.
        Obstacles list format:
          {"type": "cylinder", "center": [x, y], "radius": r, "z_min": 0, "z_max": 4, "color": (r, g, b)}
          {"type": "sphere", "center": [x, y, z], "radius": r, "color": (r, g, b)}
        """
        if isinstance(drone_euler, (float, int, np.floating)):
            euler = np.array([0.0, 0.0, float(drone_euler)], dtype=np.float64)
        else:
            euler = np.asarray(drone_euler, dtype=np.float64)

        # Initialize background based on lighting
        light_scalars = {
            "day": (1.0, 1.0, 1.0),
            "overcast": (0.65, 0.70, 0.75),
            "dim": (0.35, 0.35, 0.40),
            "dark": (0.12, 0.12, 0.15)
        }
        l_mult = light_scalars.get(lighting.lower(), (1.0, 1.0, 1.0))

        # Base background gradient (sky / floor horizon)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        horizon_y = int(self.cy - np.tan(euler[1]) * self.fy)
        horizon_y = np.clip(horizon_y, 0, self.height)

        # Sky
        sky_color = np.array([int(180 * l_mult[0]), int(210 * l_mult[1]), int(240 * l_mult[2])], dtype=np.uint8)
        frame[:horizon_y, :] = sky_color

        # Ground
        ground_color = np.array([int(90 * l_mult[0]), int(110 * l_mult[1]), int(85 * l_mult[2])], dtype=np.uint8)
        frame[horizon_y:, :] = ground_color

        depth = np.full((self.height, self.width), self.max_dist, dtype=np.float32)

        # Raycast / project obstacles sorted by distance (painter's algorithm with z-buffer check)
        obs_with_dist = []
        for obs in obstacles:
            c = np.array(obs.get("center", [0, 0, 0]), dtype=np.float64)
            if c.size == 2:
                c = np.array([c[0], c[1], drone_pos[2]])
            dist = np.linalg.norm(c - drone_pos)
            obs_with_dist.append((dist, obs))

        # Far to near
        obs_with_dist.sort(key=lambda x: x[0], reverse=True)

        for dist, obs in obs_with_dist:
            obs_type = obs.get("type", "cylinder")
            base_col = np.array(obs.get("color", (40, 40, 40)), dtype=np.float32)
            col = tuple(np.clip(base_col * np.array(l_mult), 0, 255).astype(int))

            if obs_type == "cylinder":
                # Cylindrical pillar: render vertical strip
                xy = np.array(obs["center"][:2], dtype=np.float64)
                r = float(obs.get("radius", 0.35))
                z_bot = float(obs.get("z_min", 0.0))
                z_top = float(obs.get("z_max", 4.0))

                # Discretize cylinder circumference
                angles = np.linspace(-np.pi / 2, np.pi / 2, 9)
                pts_world = []
                for a in angles:
                    perp = np.array([-np.sin(euler[2] + a), np.cos(euler[2] + a)])
                    edge_pt = xy + perp * r
                    pts_world.append(np.array([edge_pt[0], edge_pt[1], z_bot]))
                    pts_world.append(np.array([edge_pt[0], edge_pt[1], z_top]))

                # Project to camera
                pts_cam = [self.world_to_camera(pw, drone_pos, euler) for pw in pts_world]

                # Filter points ahead of camera
                valid_cam = [pc for pc in pts_cam if pc[2] > self.min_dist]
                if len(valid_cam) < 2:
                    continue

                min_z = min(pc[2] for pc in valid_cam)
                if min_z > self.max_dist:
                    continue

                # Compute 2D pixel bounds
                u_coords = [int(self.cx + (pc[0] / pc[2]) * self.fx) for pc in valid_cam]
                v_coords = [int(self.cy + (pc[1] / pc[2]) * self.fy) for pc in valid_cam]

                u_min, u_max = max(0, min(u_coords)), min(self.width, max(u_coords))
                v_min, v_max = max(0, min(v_coords)), min(self.height, max(v_coords))

                if u_max > u_min and v_max > v_min:
                    mask = depth[v_min:v_max, u_min:u_max] > min_z
                    depth[v_min:v_max, u_min:u_max][mask] = min_z
                    frame[v_min:v_max, u_min:u_max][mask] = col

            elif obs_type == "sphere":
                c_world = np.array(obs["center"], dtype=np.float64)
                r = float(obs.get("radius", 0.30))
                p_cam = self.world_to_camera(c_world, drone_pos, drone_euler)

                if p_cam[2] <= self.min_dist or p_cam[2] > self.max_dist:
                    continue

                u_c = int(self.cx + (p_cam[0] / p_cam[2]) * self.fx)
                v_c = int(self.cy + (p_cam[1] / p_cam[2]) * self.fy)
                r_pix = max(2, int((r / p_cam[2]) * self.fx))

                # Draw circle on canvas
                cv2.circle(frame, (u_c, v_c), r_pix, col, -1)
                # Update depth circular region
                y, x = np.ogrid[:self.height, :self.width]
                circle_mask = (x - u_c)**2 + (y - v_c)**2 <= r_pix**2
                depth_mask = circle_mask & (depth > p_cam[2])
                depth[depth_mask] = float(p_cam[2])

        # Dark lighting noise injection
        if lighting.lower() == "dark":
            noise = np.random.normal(0, 12, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return frame, depth
