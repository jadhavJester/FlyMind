"""
FlyDrones 3D Interactive Web Simulator Launcher.

Serves the standalone WebGL/Three.js FlyDrones application:
- 3D bedroom environment with simulated quadcopter and Drosophila brain pilot
- Fly-eye optic-flow & looming retina visualization
- Spiking neural network (850 LIF neurons) & descending neuron read-out
- Interactive swat game ('S' key), hand gesture illusions ('1'-'6'), neuron stimulation
- Webcam hand tracking integration
"""

import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "upstream", "FlyDrones", "docs"))

class FlyDronesHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DOCS_DIR, **kwargs)

    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[FlyDrones Web] {self.address_string()} - {format % args}\n")

def main():
    port = 8088
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    server_address = ("", port)
    httpd = HTTPServer(server_address, FlyDronesHTTPHandler)
    url = f"http://localhost:{port}"

    print("=" * 65)
    print(" FlyDrones — 3D Connectome Drone Web Simulator")
    print("=" * 65)
    print(f" Serving interactive WebGL simulation from:")
    print(f"   {DOCS_DIR}")
    print(f"\n >>> Open your browser at: {url}")
    print(" Press Ctrl+C to terminate.")
    print("=" * 65)

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[FlyDrones Web] Shutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    main()
