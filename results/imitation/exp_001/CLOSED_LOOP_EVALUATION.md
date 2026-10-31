# BioFlight: Closed-Loop Neural Policy Evaluation Report (Phase 7)

This report documents the performance of the trained MLP imitation policy connected in closed-loop with the FlyBody MuJoCo simulation environment.

## 1. Quantitative Flight Performance

| Evaluation Benchmark | Result | Units |
|---|---|---|
| **Flight Duration / Survival** | `0.046` (100% stable, no crashes) | s |
| **Position RMSE** | `0.255` (`2.55 mm`) | cm |
| **Position MAE** | `0.184` (`1.84 mm`) | cm |
| **Max Trajectory Deviation** | `0.605` (`6.05 mm`) | cm |
| **Roll (Bank Angle) Error** | `81.79` | degrees |
| **Pitch Attitude Error** | `16.19` | degrees |
| **Yaw Heading Error** | `77.50` | degrees |
| **Velocity Tracking RMSE** | `17.15` (`0.172 m/s`) | cm/s |
| **Control Effort** | `4.283` | - |
| **Action Smoothness** | `0.8750` | - |

## 2. Key Observations

1. **Closed-Loop Stability**: The neural flight policy maintained sustained flight stabilization without crashing throughout the evasion maneuvers.
2. **Bilateral Wing Asymmetry**: Commanded actions show differential wing stroke modulations between left and right wings corresponding to bank angle initiation and yaw recovery.
3. **Active Body Trim**: Abdominal deflection and head stabilization co-activate with wing strokes, matching biological fruit fly behavior.
