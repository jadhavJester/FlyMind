"""
FlyMind UE5 Mock Client.

Simulates an Unreal Engine 5 game viewport client streaming camera frames
over TCP socket to the FlyMind Biological Intelligence Bridge Server.
"""
import socket
import struct
import time
import numpy as np

from ..sensors.synthetic_camera import SyntheticCamera
from ..environments.dynamic_looming import DynamicLoomingArena


def run_mock_client(
    host: str = "127.0.0.1",
    port: int = 9876,
    fps: int = 50,
    duration_s: float = 10.0
) -> None:
    """Connects to FlyMind Bridge Server and streams simulated frames."""
    cam = SyntheticCamera(width=160, height=120, hfov_deg=80.0)
    arena = DynamicLoomingArena(approach_speed=2.5, initial_distance=12.0, hazard_radius=0.45)

    print(f"[Mock Client] Connecting to FlyMind Bridge at {host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
        print("[Mock Client] Connected successfully! Streaming camera frames...")
    except ConnectionRefusedError:
        print(f"[Mock Client] Could not connect to {host}:{port}. Ensure UE5BridgeServer is running.")
        return

    drone_pos = np.array([0.0, 0.0, 1.5], dtype=np.float64)
    drone_euler = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    start_time = time.perf_counter()
    dt = 1.0 / fps

    try:
        step = 0
        while time.perf_counter() - start_time < duration_s:
            t_now = time.perf_counter() - start_time
            obstacles = arena.get_obstacles(t_now)
            frame, _ = cam.render(drone_pos, drone_euler, obstacles)

            # Ensure BGR uint8 contiguous bytes
            h, w, c = frame.shape
            raw_bytes = frame.tobytes()

            # Pack 16-byte header: Magic 'FLYM', Width, Height, Timestamp_ms
            header = struct.pack("!4sIII", b"FLYM", w, h, int(t_now * 1000))

            sock.sendall(header + raw_bytes)
            step += 1
            time.sleep(dt)

        print(f"[Mock Client] Streamed {step} frames ({duration_s:.1f}s) cleanly.")
    except (ConnectionResetError, BrokenPipeError) as e:
        print(f"[Mock Client] Server closed connection ({e}).")
    finally:
        sock.close()


if __name__ == "__main__":
    run_mock_client()
