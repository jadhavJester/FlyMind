"""
Diagnostic script to inspect the generated backflip telemetry and video consistency.
"""
import json
import numpy as np

def inspect():
    with open('results/backflip/backflip_telemetry.json', 'r') as f:
        data = json.load(f)
    t = data['telemetry']
    
    print(f"Total steps: {len(t['step'])}")
    print(f"Max pitch rotation: {data['max_pitch_rotation_deg']:.1f}°")
    print(f"Final pitch rotation: {data['final_pitch_rotation_deg']:.1f}°")
    print(f"Altitude range: min={min(t['pos_z_cm']):.2f} cm, max={max(t['pos_z_cm']):.2f} cm, final={t['pos_z_cm'][-1]:.2f} cm")
    print(f"Lateral Y drift range: min={min(t['pos_y_cm']):.2f} cm, max={max(t['pos_y_cm']):.2f} cm")
    
    print("\nStep-by-step key frames:")
    for i in range(0, len(t['step']), 15):
        ph = t['phase'][i]
        x = t['pos_x_cm'][i]
        y = t['pos_y_cm'][i]
        z = t['pos_z_cm'][i]
        fx = t['fwd_x'][i]
        fz = t['fwd_z'][i]
        uz = t['up_z'][i]
        rot = t['continuous_pitch_deg'][i]
        pr = t['pitch_rate_degs'][i]
        print(f"Step {i:3d} [{ph:<17}]: Pos=({x:5.2f}, {y:5.2f}, {z:5.2f}) | Fwd=({fx:5.2f}, {fz:5.2f}) | Up_Z={uz:5.2f} | Rot={rot:5.1f}° | Rate={pr:7.0f}°/s")

if __name__ == "__main__":
    inspect()
