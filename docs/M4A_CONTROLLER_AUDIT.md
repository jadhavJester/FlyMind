# Milestone M4A — Controller Audit: Drosophila Flight Policy & Operating Interface

**Document Version**: 1.0  
**Date**: October 3, 2026  
**Status**: APPROVED AUDIT REPORT  
**Scope**: FlyBody learned flight controller, state observation mapping, trajectory generator, and long-range failure analysis.

---

## Executive Summary

This audit rigorously inspects the flight control pipeline utilized in Milestone 3 ("Drosophila Arithmetic Selector v0.1") and the Alphabet Spelling task. The goal is to determine the architectural cause of the **long-range flight failure**: while the fly reliably navigates to short targets ($< 1.0\text{ cm}$), maneuvers spanning several centimeters lead to systematic altitude decay and eventual floor contact.

The investigation confirms that:
1. The neural flight policy operates on an ad-hoc 28-dimensional kinematic/lookahead vector rather than the native 104-dimensional FlyBody observation space.
2. The reference trajectory is computed **once at episode reset** as a static open-loop line and is never dynamically updated or replanned during flight.
3. The underlying neural model (`checkpoints/imitation/exp_001/best_model.pt`) was trained via behavioral cloning strictly on brief saccades and evasions ($50 - 200$ frames $\approx 10 - 40\text{ ms}$, total path $< 1.5\text{ cm}$).
4. There is **zero active altitude feedback, trim compensation, or closed-loop waypoint replanning**. When tracking long trajectories, accumulated aerodynamic drag and vertical tracking drift cause uncorrected altitude loss.

---

## 1. Where the 28-Dimensional Controller Input is Created

The 28-dimensional feature vector is created at the environment-policy interface across three core files:
1. **[src/environments/interactive_flight.py](file:///d:/Projects/Fruit%20fly/src/environments/interactive_flight.py#L190-L226)**: In `get_observation()`, lines 190–226.
2. **[src/data/preprocessor.py](file:///d:/Projects/Fruit%20fly/src/data/preprocessor.py#L116-L168)**: In `_extract_features()`, lines 116–168, during offline dataset processing.
3. **[scripts/run_web_control_center.py](file:///d:/Projects/Fruit%20fly/scripts/run_web_control_center.py#L250-L280)**: In `step_simulation()`, lines 250–280, for the live control center loop.

At every control step ($dt = 0.2\text{ ms}$), the simulator queries MuJoCo physics for instant kinematics and computes forward displacements to 5 future reference points along the active trajectory generator.

---

## 2. Representation of the 28 Observation Dimensions

The 28-dimensional input vector $\mathbf{x} \in \mathbb{R}^{28}$ is composed of:

| Dimension Index | Channel Name | Representation / Physics | Units | Frame / Source |
| :---: | :---: | :---: | :---: | :---: |
| **0 – 2** | `fly_lin_vel` | Body linear velocity $[v_x, v_y, v_z]$ | $\text{cm/s}$ | Thorax center-of-mass (`physics.data.qvel[:3]`) |
| **3 – 5** | `fly_ang_vel` | Body angular velocity $[\omega_x, \omega_y, \omega_z]$ | $\text{rad/s}$ | Rotational rate (`physics.data.qvel[3:6]`) |
| **6 – 9** | `fly_quat` | Attitude unit quaternion $[q_w, q_x, q_y, q_z]$ | Dimensionless | MuJoCo root joint attitude (`physics.data.qpos[3:7]`) |
| **10 – 12** | `fly_euler` | Attitude Euler angles $[\phi_{\text{roll}}, \theta_{\text{pitch}}, \psi_{\text{yaw}}]$ | $\text{rad}$ | Analytical conversion from quaternion |
| **13 – 15** | `future_disp_1` | Relative vector $(p_{t+1} - p_t)$ to 1st lookahead waypoint | $\text{cm}$ | Reference path error at $+0.2\text{ ms}$ |
| **16 – 18** | `future_disp_2` | Relative vector $(p_{t+2} - p_t)$ to 2nd lookahead waypoint | $\text{cm}$ | Reference path error at $+0.4\text{ ms}$ |
| **19 – 21** | `future_disp_3` | Relative vector $(p_{t+3} - p_t)$ to 3rd lookahead waypoint | $\text{cm}$ | Reference path error at $+0.6\text{ ms}$ |
| **22 – 24** | `future_disp_4` | Relative vector $(p_{t+4} - p_t)$ to 4th lookahead waypoint | $\text{cm}$ | Reference path error at $+0.8\text{ ms}$ |
| **25 – 27** | `future_disp_5` | Relative vector $(p_{t+5} - p_t)$ to 5th lookahead waypoint | $\text{cm}$ | Reference path error at $+1.0\text{ ms}$ |

---

## 3. Relationship to Native 104-Dimensional FlyBody Observation

**The 28 dimensions are NOT derived from the native 104-dimensional FlyBody observation dictionary.**

In native FlyBody, the task environment (`flight_imitation`) produces an observation dictionary comprising 104 scalar features (detailed in [docs/observation_space.md](file:///d:/Projects/Fruit%20fly/docs/observation_space.md)):
- Wing joint angles (yaw, roll, pitch for both wings): 6 dims
- Wing joint velocities: 6 dims
- Head and abdomen joint angles & velocities: 10 dims
- Actuator forces: 12 dims
- Fly thorax linear/angular velocities in body frame: 6 dims
- Fly orientation (quaternion): 4 dims
- Reference trajectory lookahead displacements (30 future steps $\times 2 = 60$ dims)

Instead of using the standard FlyBody observation structure, the project's current policy bypasses the native observation builder and constructs its own custom 28-dimensional vector directly from MuJoCo raw state `physics.data.qpos` and `physics.data.qvel`.

---

## 4. Native Observation Dimensions Actually Utilized

Comparing the native 104 FlyBody observation channels with the 28 features used:

1. **Utilized (or equivalent)**:
   - Root linear velocity: 3 dims (world frame rather than body frame)
   - Root angular velocity: 3 dims
   - Root quaternion: 4 dims
   - Future trajectory displacements: 15 dims (5 steps $\times 3$ coordinates, vs. FlyBody's default $30$ steps)
2. **Completely Ignored / Excluded**:
   - All wing joint kinematics (6 angles, 6 velocities)
   - All actuator forces (12 dimensions)
   - All head and abdominal state variables (10 dimensions)
   - Internal WBPG (Wing Beat Pattern Generator) phase angles and lift bias
   - Ground contact and collision sensors

The policy therefore acts strictly on macroscopic rigid-body kinematics and 5 lookahead errors, with zero internal proprioceptive awareness of wing dynamics.

---

## 5. Presence of Privileged Environment Information

**YES, privileged ground-truth reference information is present:**
1. **Lookahead Displacements ($H=5$)**: The policy directly receives the exact Euclidean error vectors $[p_{ref}(t+k) - p_{fly}(t)]$ for $k \in \{1, 2, 3, 4, 5\}$.
2. **Terminal Target Substitution**: When the reference trajectory ends (step count $> N_{traj}$), lines 215–217 of `interactive_flight.py` explicitly substitute the exact 3D world coordinates of the selected target into the lookahead slots:
   ```python
   ref_point = self.targets[self.selected_target_id].position
   future_disp.extend(ref_point - fly_pos)
   ```
This provides perfect global coordinate knowledge without any visual, sensory, or odometric estimation noise.

---

## 6. Trained Model / Checkpoint Generating the 12 Actions

- **Checkpoint File**: `checkpoints/imitation/exp_001/best_model.pt`
- **Metadata File**: `data/processed/metadata.json` (defines mean and std vectors for z-score normalization)
- **Architecture**: `MLPPolicy` ([src/models/mlp_policy.py](file:///d:/Projects/Fruit%20fly/src/models/mlp_policy.py))
  - Input Layer: `LayerNorm(28)`
  - Hidden Layers: `Linear(28, 256) -> SiLU -> Linear(256, 256) -> SiLU -> Linear(256, 128) -> SiLU`
  - Output Head: `Linear(128, 12) -> tanh * 3.0`
  - Action Space: Continuous $\mathbb{R}^{12}$, bounded in $[-3.0, +3.0]$, driving wing pitch/yaw/roll differentials, head motion, abdominal posture, and wing-beat amplitude bias.

---

## 7. How the Reference Trajectory is Generated

The reference trajectory is generated via `flybody.tasks.synthetic_trajectories.constant_speed_trajectory(...)`:
```python
heading = float(np.arctan2(target_pos[1], target_pos[0]))
qp, qv = constant_speed_trajectory(
    n_steps=400,
    speed=20.0,
    init_pos=(0.0, 0.0, 1.037),
    init_heading=heading,
    body_rot_angle_y=-47.5,
    control_timestep=0.0002
)
self.underlying_env.task._traj_generator.set_next_trajectory(qp, qv)
```
- It assumes a single straight line from the initial fly spawn point $(0, 0, 1.037\text{ cm})$ along the horizontal azimuth toward the target.
- Altitude is constant ($z(t) \equiv 1.037\text{ cm}$).
- Pitch angle is set to a constant physiological cruise angle ($\theta_{pitch} = -47.5^\circ$).

---

## 8. Frequency of Trajectory Updates

**The reference trajectory is updated exactly ONCE per episode (at episode reset).**

During simulation execution, the trajectory is completely static. The flight controller operates open-loop relative to this pre-generated spline. It does NOT adjust for drift, external disturbances, heading misalignment, or accumulated vertical drop.

---

## 9. Altitude Stabilization / Recovery Mechanisms

**There is NO explicit altitude stabilization or recovery mechanism.**
- The neural policy is a memoryless feed-forward network ($28 \to 12$).
- The loss function during training was standard Mean Squared Error against short biological evasion data.
- The simulator has no vertical integrator, no PID altitude hold, and no pitch adjustment loop.
- If aerodynamic lift fails to match the fly's weight during a maneuver, vertical velocity $v_z$ becomes negative. The policy receives $v_z < 0$ and $z_{ref} - z > 0$, but because it was never trained to perform sustained climbs from deep altitude deficits, it fails to generate sufficient wing beat amplitude to arrest the fall.

---

## 10. Training Distribution: Trajectory Durations & Distances

**The current policy was trained strictly on short-horizon saccades and evasive bursts.**
- **Source Data**: High-speed multi-camera Drosophila flight tracking dataset (`data/raw/flight_trajectories.h5`).
- **Duration**: Episodes in the dataset range from 50 to 200 time steps ($10\text{ ms} - 40\text{ ms}$).
- **Spatial Scale**: The total displacement of trajectories in the training set is typically **$0.4\text{ cm} - 1.2\text{ cm}$**.
- **Long-Range Flight**: The policy was never exposed to flights longer than $\sim 1.5\text{ cm}$ or durations exceeding $\sim 0.08\text{ seconds}$.
- **Consequence**: When commanded to fly $3.0\text{ cm}$ to $8.0\text{ cm}$ across the arena, the controller operates far outside its support distribution, accumulating tracking error and losing altitude until touchdown.

---

## Conclusion: Architectural Roadmap for M4A

To fix long-range flight without retraining the neural network:
1. **Empirical Envelope Characterization**: Measure the exact distance ($d_{safe}$) and heading angle ($\Delta \psi_{safe}$) over which the neural controller maintains altitude.
2. **Local Waypoint Planner**: Decompose multi-centimeter targets into a succession of safe local sub-waypoints ($\le d_{safe}$) generated iteratively closed-loop.
3. **Active Altitude Supervisor**: Monitor instantaneous altitude $z_{fly}$; trigger an altitude preservation/climb maneuver whenever altitude falls below a defined safety floor before executing further lateral displacement.
