# Experimental Results Log: FlyBody Interactive Agent Experiment

## Milestone Overview & Progress
* **Milestone 0 (Video Analysis)**: Complete ([`docs/video_analysis.md`](file:///d:/Projects/Fruit%20fly/docs/video_analysis.md)).
* **Milestone 1 (Official Simulation Verification)**: Complete ([`docs/milestone_01.md`](file:///d:/Projects/Fruit%20fly/docs/milestone_01.md)).
* **Milestone 2 (Agent Interface Inspection & Controllability)**: Complete ([`docs/milestone_02_source_trace.md`](file:///d:/Projects/Fruit%20fly/docs/milestone_02_source_trace.md), [`docs/observation_space.md`](file:///d:/Projects/Fruit%20fly/docs/observation_space.md), [`docs/action_space.md`](file:///d:/Projects/Fruit%20fly/docs/action_space.md), [`docs/control_pipeline.md`](file:///d:/Projects/Fruit%20fly/docs/control_pipeline.md)).

---

## 1. Milestone 1 Results: Physics & Locomotion Verification
* **Objective**: Validate the official FlyBody simulation pipeline on local CPU hardware without CUDA.
* **Outcome**: Verified end-to-end physics integration at 20,000 Hz, control at 5,000 Hz, active closed-loop neural controller, and genuine physical displacement.
* **Key Metrics**:
  * FlyBody Version: 0.1.0 (MuJoCo 3.13.0, dm_control 1.0.46)
  * Net 3D Displacement: $0.3070\text{ cm}$ ($3.07\text{ mm}$)
  * Flight Speed: $24.99\text{ cm/s}$ ($0.25\text{ m/s}$)
  * Control Frequency: 5,000 Hz / Physics Frequency: 20,000 Hz
* **Artifacts**:
  * Screenshot: [`results/milestone_01/milestone_01_frame.png`](file:///d:/Projects/Fruit%20fly/results/milestone_01/milestone_01_frame.png)
  * Video (MP4): [`videos/milestone_01_simulation.mp4`](file:///d:/Projects/Fruit%20fly/videos/milestone_01_simulation.mp4)
  * Video (WebP): [`videos/milestone_01_simulation.webp`](file:///d:/Projects/Fruit%20fly/videos/milestone_01_simulation.webp)

---

## 2. Milestone 2 Results: Interface Inspection & Controllability Analysis
* **Objective**: Fully characterize what the agent observes, what it controls, and empirically measure flight dynamics under controlled action perturbations.
* **Inspection Script**: [`scripts/inspect_flybody_interface.py`](file:///d:/Projects/Fruit%20fly/scripts/inspect_flybody_interface.py)

### Observation Space Summary
* **Total Dimension**: **104 continuous numerical features** (`float64`).
* **Composition**:
  * `accelerometer` (3): Specific force in thorax frame.
  * `gyro` (3): Angular rates in thorax frame.
  * `joints_pos` (25): Hinge angles across active flight and steering joints.
  * `joints_vel` (25): Hinge angular velocities.
  * `velocimeter` (3): Linear velocities in thorax frame.
  * `world_zaxis` (3): World vertical vector in thorax frame (gravitational tilt).
  * `ref_displacement` (18): 6 lookahead 3D relative target vectors in egocentric coordinates.
  * `ref_root_quat` (24): 6 lookahead relative orientation quaternions in local frame.
* **Finding**: The agent receives **purely egocentric sensory and relative tracking data**, with **zero absolute world coordinates**.

### Action Space Summary
* **Total Channels**: **12 continuous channels** (`float64`).
* **Composition**:
  * Channels 0..2: Head joints (abduct, twist, pitch) $\to$ direct PD position targets.
  * Channels 3..8: Wings (yaw, roll, pitch for left & right) $\to$ kinematic stroke offsets added to 218 Hz WBPG baseline.
  * Channels 9..10: Abdomen (abduct, pitch) $\to$ direct PD position targets (aerodynamic rudder).
  * Channel 11: WBPG frequency bias $\to$ scales flapping frequency $\pm 10\%$.

### Action Perturbation Controllability Results
Empirical responses over 150 control timesteps ($30\text{ ms}$):

| Experiment | Action Configuration | Net Displacement | Final Speed | Final Euler Attitude [Roll, Pitch, Yaw] | Physical Interpretation |
|---|---|---|---|---|---|
| **Baseline** | Zero bias ($\mathbf{a} = \mathbf{0}$) | $0.598\text{ cm}$ | $40.98\text{ cm/s}$ | $[-42.3^\circ, -38.0^\circ, -71.9^\circ]$ | Forward-descending baseline glide/cruise |
| **Differential Wing Roll** | $a_4 = +0.5, a_7 = -0.5$ | $0.573\text{ cm}$ | $34.48\text{ cm/s}$ | $[-60.3^\circ, -39.9^\circ, +152.5^\circ]$ | Generates rapid coordinated bank/roll torque ($> 60^\circ$ roll) |
| **Symmetric Wing Pitch** | $a_5 = +0.5, a_8 = +0.5$ | $0.437\text{ cm}$ | $15.36\text{ cm/s}$ | $[+139.8^\circ, +6.8^\circ, -167.1^\circ]$ | Increases angle of attack, shifts thrust vector, decelerates speed |
| **Flapping Frequency Boost** | $a_{11} = +1.0$ (Max freq) | $0.520\text{ cm}$ | $10.17\text{ cm/s}$ | $[-123.3^\circ, -43.6^\circ, +26.0^\circ]$ | Maximizes vertical aerodynamic lift and alters pitch trim |
| **Abdominal Lateral Steering** | $a_9 = +0.5$ | $0.601\text{ cm}$ | $20.83\text{ cm/s}$ | $[-132.4^\circ, +86.1^\circ, -88.4^\circ]$ | High yaw/pitch trimming via biological rudder |

---

## 4. Milestone 3 Results: Drosophila Arithmetic Selector v0.1
* **Objective**: Establish the first complete embodied closed-loop task: Task $\to$ Decision $\to$ Target $\to$ Flight $\to$ Contact $\to$ Reward.
* **Task**: $2 + 3 = ?$ with candidate answers $[5]$ (Left) and $[7]$ (Right).
* **Script**: [`scripts/run_milestone_03.py`](file:///d:/Projects/Fruit%20fly/scripts/run_milestone_03.py)
* **Documentation**: [`docs/milestone_03.md`](file:///d:/Projects/Fruit%20fly/docs/milestone_03.md)

### Quantitative Summary:
| Baseline Mode | Commanded Target | Target Value | Steps | Flight Duration | Final Pos $[X, Y, Z]\text{ cm}$ | Min Dist to [5] | Min Dist to [7] | Reward | Termination | Success |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A (Oracle)** | `target_left` | 5 | 183 | $36.6\text{ ms}$ | $[0.609, 0.252, 0.401]$ | **$0.1495\text{ cm}$** | $0.4417\text{ cm}$ | **$+8.18$** | `CORRECT_TARGET_CONTACT` | **YES** |
| **Baseline B (Wrong)** | `target_right` | 7 | 182 | $36.4\text{ ms}$ | $[0.597, -0.251, 0.401]$ | $0.5645\text{ cm}$ | **$0.1493\text{ cm}$** | **$-11.81$** | `WRONG_TARGET_CONTACT` | **NO** |
| **Baseline C (Solver)** | `target_left` | 5 | 190 | $38.0\text{ ms}$ | $[0.627, 0.256, 0.400]$ | **$0.1497\text{ cm}$** | $0.4887\text{ cm}$ | **$+8.11$** | `CORRECT_TARGET_CONTACT` | **YES** |

### Artifacts:
* **Trajectory Comparison Plot**: [`results/milestone_03/trajectory_comparison.png`](file:///d:/Projects/Fruit%20fly/results/milestone_03/trajectory_comparison.png)
* **Telemetry Video (MP4)**: [`videos/milestone_03_arithmetic.mp4`](file:///d:/Projects/Fruit%20fly/videos/milestone_03_arithmetic.mp4)
* **Telemetry Animation (WebP)**: [`videos/milestone_03_arithmetic.webp`](file:///d:/Projects/Fruit%20fly/videos/milestone_03_arithmetic.webp)
* **Episode JSON Logs**: [`results/milestone_03/episode_logs.json`](file:///d:/Projects/Fruit%20fly/results/milestone_03/episode_logs.json)
* **Summary JSON**: [`results/milestone_03/summary.json`](file:///d:/Projects/Fruit%20fly/results/milestone_03/summary.json)
