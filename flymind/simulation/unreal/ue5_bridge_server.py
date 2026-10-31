"""
FlyMind Unreal Engine 5 Real-Time Communication Protocol & Bridge Server.

Listens for incoming camera streams and IMU telemetry from Unreal Engine 5,
runs the real-time insect-inspired visual perception pipeline (EMD, LPLC2, Flow Divergence),
and renders a comprehensive research HUD.
"""
import socket
import struct
import time
from typing import Optional, Tuple, Any
import cv2
import numpy as np

from ...perception.optical_flow.farneback_dense import FarnebackDenseFlow
from ...perception.elementary_motion_detection.hr_emd import HRElementaryMotionDetector
from ...perception.nearness_estimation.flow_divergence import FlowDivergenceNearness
from ...bioinspired.lplc2.population_coding import LPLC2Population
from ...bioinspired.neural_models.giant_fiber_lif import GiantFiberNeuron


class UE5BridgeServer:
    """
    High-performance TCP Socket Server for streaming Unreal Engine 5 camera frames
    directly into FlyMind's biological intelligence pipeline.
    
    Wire Protocol:
        [Header: 16 bytes]
            - Magic: 4 bytes (ASCII 'FLYM')
            - Width: uint32
            - Height: uint32
            - Timestamp_ms: uint32
        [Payload: Width * Height * 3 bytes (BGR/RGB raw bytes) or JPEG encoded]
    """
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9876,
        frame_width: int = 160,
        frame_height: int = 120,
    ):
        self.host = host
        self.port = port
        self.width = frame_width
        self.height = frame_height

        # Biological Perception Pipeline
        self.farneback = FarnebackDenseFlow(pyr_scale=0.5, levels=2, winsize=11, iterations=2)
        self.emd = HRElementaryMotionDetector(tau=0.04, dt=0.02)
        self.lplc2 = LPLC2Population(image_width=self.width, image_height=self.height)
        self.gf = GiantFiberNeuron(v_thresh_mv=-45.0, weight_lplc2=3.0)
        self.nearness = FlowDivergenceNearness()

        self._running = False
        self.sock: Optional[socket.socket] = None

    def start(self, display_hud: bool = True) -> None:
        """Starts the server loop and processes frames in real-time."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        self._running = True

        print(f"\n=======================================================")
        print(f" FlyMind UE5 Bridge Server listening on {self.host}:{self.port}")
        print(f" Waiting for Unreal Engine 5 client connection...")
        print(f"=======================================================\n")

        try:
            while self._running:
                client_sock, client_addr = self.sock.accept()
                print(f"[UE5 Bridge] Connected to client: {client_addr}")
                self._handle_client(client_sock, display_hud=display_hud)
        except KeyboardInterrupt:
            print("\n[UE5 Bridge] Server stopped by user.")
        finally:
            self.stop()

    def _handle_client(self, client_sock: socket.socket, display_hud: bool = True) -> None:
        """Receives and processes incoming frames from connected client."""
        client_sock.settimeout(5.0)
        last_time = time.perf_counter()

        try:
            while self._running:
                # 1. Read Header (16 bytes)
                header_data = self._recv_exact(client_sock, 16)
                if not header_data:
                    break

                magic, w, h, ts_ms = struct.unpack("!4sIII", header_data)
                if magic != b"FLYM":
                    print(f"[UE5 Bridge] Invalid magic bytes: {magic}")
                    break

                # 2. Read Frame Payload
                payload_len = w * h * 3
                raw_bytes = self._recv_exact(client_sock, payload_len)
                if not raw_bytes:
                    break

                # Convert to numpy frame
                frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 3))
                current_time = time.perf_counter()
                dt = max(0.001, current_time - last_time)
                last_time = current_time

                # 3. Process through Biological Intelligence Pipeline
                hud = self.process_frame(frame, dt=dt)

                # 4. Optional: Send back navigation setpoint (Twist command)
                # cmd_vel_x, cmd_vel_y, cmd_vel_z, cmd_yaw_rate
                # client_sock.sendall(struct.pack("!ffff", vx, vy, vz, yaw_rate))

                # 5. Display Research HUD
                if display_hud:
                    cv2.imshow("FlyMind: Insect Visual Intelligence (UE5 Bridge)", hud)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):
                        self._running = False
                        break

        except (socket.timeout, ConnectionResetError) as e:
            print(f"[UE5 Bridge] Client disconnected ({e}).")
        finally:
            client_sock.close()
            if display_hud:
                cv2.destroyAllWindows()

    def process_frame(self, frame_bgr: np.ndarray, dt: float = 0.02) -> np.ndarray:
        """
        Executes the biological perception pipeline on a single frame
        and synthesizes a 4-quadrant research HUD.
        """
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]

        # 1. Optical Flow & Divergence Nearness
        flow_u, flow_v = self.farneback.compute(frame_bgr)
        nearness_map = self.nearness.compute(flow_u, flow_v, forward_speed=1.5, dt=dt)
        nearness_norm = np.clip(nearness_map / self.nearness.max_nearness * 255.0, 0, 255).astype(np.uint8)
        nearness_vis = cv2.applyColorMap(nearness_norm, cv2.COLORMAP_TURBO)

        # 2. Elementary Motion Detection (T4/T5 channels)
        emd_channels = self.emd.process_frame(frame_bgr)

        # 3. LPLC2 Multiplicative Cardinal Cross Integration
        lplc2_resp = self.lplc2.process(emd_channels)

        # 4. Giant Fiber Invariant Looming Escape Trigger
        gf_state = self.gf.step(lplc2_resp.active_count, lplc2_resp.mean_activation, dt=dt)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Construct Research HUD
        hud = self._build_hud(
            frame_bgr=frame_bgr,
            nearness_vis=nearness_vis,
            emd_channels=emd_channels,
            lplc2_resp=lplc2_resp,
            gf_state=gf_state,
            latency_ms=latency_ms,
        )
        return hud

    def _build_hud(
        self,
        frame_bgr: np.ndarray,
        nearness_vis: np.ndarray,
        emd_channels: dict,
        lplc2_resp: Any,
        gf_state: Any,
        latency_ms: float
    ) -> np.ndarray:
        """Combines perception layers into a 4-quadrant research dashboard."""
        h, w = frame_bgr.shape[:2]

        # Canvas: 2x2 grid, each cell (w, h)
        hud = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

        # Quadrant 1 (Top-Left): Input Camera View
        q1 = frame_bgr.copy()
        cv2.putText(q1, "UE5 Camera Stream", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        hud[0:h, 0:w] = q1

        # Quadrant 2 (Top-Right): Relative Nearness Proximity Heatmap
        q2 = cv2.resize(nearness_vis, (w, h))
        cv2.putText(q2, "Relative Nearness Map", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        hud[0:h, w:w*2] = q2

        # Quadrant 3 (Bottom-Left): T4/T5 4-Channel Cardinal Motion
        q3 = np.zeros((h, w, 3), dtype=np.uint8)
        right_norm = np.clip(emd_channels.get("right", np.zeros((h, w))) * 255, 0, 255).astype(np.uint8)
        left_norm = np.clip(emd_channels.get("left", np.zeros((h, w))) * 255, 0, 255).astype(np.uint8)
        q3[:, :, 2] = right_norm  # Red for rightward motion
        q3[:, :, 0] = left_norm   # Blue for leftward motion
        cv2.putText(q3, "T4/T5 Motion Detectors (L/R)", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        hud[h:h*2, 0:w] = q3

        # Quadrant 4 (Bottom-Right): LPLC2 Looming & Giant Fiber Telemetry
        q4 = np.zeros((h, w, 3), dtype=np.uint8)
        # Status header
        status_color = (0, 0, 255) if gf_state.is_spiking else (0, 255, 0)
        status_text = "GIANT FIBER: ESCAPE SACCADE!" if gf_state.is_spiking else "GIANT FIBER: SAFE CRUISE"
        cv2.putText(q4, status_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

        cv2.putText(q4, f"LPLC2 Active Units: {lplc2_resp.active_count}", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(q4, f"LPLC2 Max Drive:   {lplc2_resp.max_activation:.3f}", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(q4, f"GF Membrane (mV):  {gf_state.membrane_potential_mv:.1f} mV", (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(q4, f"Bio Latency:       {latency_ms:.1f} ms (~{1000.0/max(0.1, latency_ms):.0f} Hz)", (8, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Draw membrane potential gauge bar
        v_pct = np.clip((gf_state.membrane_potential_mv - (-65.0)) / ((-45.0) - (-65.0)), 0.0, 1.0)
        bar_w = int((w - 16) * v_pct)
        cv2.rectangle(q4, (8, 104), (w - 8, 114), (50, 50, 50), -1)
        cv2.rectangle(q4, (8, 104), (8 + bar_w, 114), status_color, -1)

        hud[h:h*2, w:w*2] = q4

        # Scale up for crisp display
        return cv2.resize(hud, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)

    def _recv_exact(self, sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        """Reads exactly num_bytes from socket stream."""
        buffer = bytearray()
        while len(buffer) < num_bytes:
            packet = sock.recv(num_bytes - len(buffer))
            if not packet:
                return None
            buffer.extend(packet)
        return bytes(buffer)

    def stop(self) -> None:
        """Closes the server socket."""
        self._running = False
        if self.sock:
            self.sock.close()
            self.sock = None
