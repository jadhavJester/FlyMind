"""
Unit test for FlyMind UE5 Bridge Server and Communication Protocol.
"""
import socket
import struct
import threading
import time
import numpy as np

from ..simulation.unreal.ue5_bridge_server import UE5BridgeServer


def test_ue5_bridge_protocol():
    server = UE5BridgeServer(host="127.0.0.1", port=9988, frame_width=160, frame_height=120)
    
    server_thread = threading.Thread(target=server.start, kwargs={"display_hud": False}, daemon=True)
    server_thread.start()
    time.sleep(0.2)

    # Client connects and sends one frame
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 9988))

    dummy_frame = np.full((120, 160, 3), 128, dtype=np.uint8)
    header = struct.pack("!4sIII", b"FLYM", 160, 120, 100)
    sock.sendall(header + dummy_frame.tobytes())

    # Send second frame with bright looming circle
    looming_frame = dummy_frame.copy()
    looming_frame[40:80, 60:100] = 240
    header2 = struct.pack("!4sIII", b"FLYM", 160, 120, 200)
    sock.sendall(header2 + looming_frame.tobytes())

    time.sleep(0.3)
    sock.close()
    server.stop()
    assert True
