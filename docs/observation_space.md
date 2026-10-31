# FlyBody Observation Space: Complete Mapping & Characterization

## 1. Overview & Verification
The observation space of FlyBody's `flight_imitation` environment was verified directly from the environment specification and live physics state.

* **Total Flattened Dimension**: **104 continuous numerical values** (`float64`).
* **Vision Flight Dimension** (`vision_guided_flight`): **6,208 values** ($32 \times 32 \times 3$ left eye + $32 \times 32 \times 3$ right eye + 62 proprioceptive/vestibular + 2 task inputs).
* **Control Rate**: Evaluated and returned at **5,000 Hz** ($\Delta t = 0.0002\text{ s}$).

---

## 2. Complete 104-Dimensional Observation Mapping Table

| Flattened Indices | Observable Name | Component | Physical Meaning | Units | Coordinate Frame | Normalization / Bounds |
|---:|---|---|---|---|---|---|
| **0 .. 2** | `walker/accelerometer` | $[a_x, a_y, a_z]$ | Thorax specific force (linear acceleration minus gravity) | $\text{cm/s}^2$ | Local Thorax Frame | Unbounded ($\approx \pm 5 \times 10^3$) |
| **N/A** | `walker/actuator_activation` | Empty (0 dims) | Actuator first-order filter activation states | $[-1, 1]$ | Actuator state | 0 dims (filter disabled in flight) |
| **3 .. 5** | `walker/gyro` | $[\omega_x, \omega_y, \omega_z]$ | Thorax angular velocity (roll, pitch, yaw rate) | $\text{rad/s}$ | Local Thorax Frame | Typically $\approx \pm 100\text{ rad/s}$ |
| **6 .. 30** | `walker/joints_pos` | 25 joint angles | Angular positions of all active joints (head, wings, abdomen, halteres) | $\text{rad}$ | Joint hinge local frame | Joint physical limits ($[-3, 3]$) |
| **31 .. 55** | `walker/joints_vel` | 25 joint rates | Angular velocities of all active joints | $\text{rad/s}$ | Joint hinge local frame | Typically $\approx \pm 500\text{ rad/s}$ |
| **56 .. 58** | `walker/velocimeter` | $[v_x, v_y, v_z]$ | Thorax linear velocity vector | $\text{cm/s}$ | Local Thorax Frame | Typically $0 \dots 100\text{ cm/s}$ |
| **59 .. 61** | `walker/world_zaxis` | $[z_x, z_y, z_z]$ | Projection of world vertical vector $[0, 0, 1]^T$ into thorax frame | Dimensionless | Local Thorax Frame | Unit sphere $\|\mathbf{z}\| = 1.0$ |
| **62 .. 64** | `walker/ref_displacement[0]` | $[\Delta x_0, \Delta y_0, \Delta z_0]$ | Vector from fly CoM to target CoM at current step ($t$) | $\text{cm}$ | Fly Egocentric Frame | Terminal threshold $2.0\text{ cm}$ |
| **65 .. 67** | `walker/ref_displacement[1]` | $[\Delta x_1, \Delta y_1, \Delta z_1]$ | Lookahead displacement to target CoM at $t + 1$ step ($+0.2\text{ ms}$) | $\text{cm}$ | Fly Egocentric Frame | Relative spatial distance |
| **68 .. 70** | `walker/ref_displacement[2]` | $[\Delta x_2, \Delta y_2, \Delta z_2]$ | Lookahead displacement to target CoM at $t + 2$ step ($+0.4\text{ ms}$) | $\text{cm}$ | Fly Egocentric Frame | Relative spatial distance |
| **71 .. 73** | `walker/ref_displacement[3]` | $[\Delta x_3, \Delta y_3, \Delta z_3]$ | Lookahead displacement to target CoM at $t + 3$ step ($+0.6\text{ ms}$) | $\text{cm}$ | Fly Egocentric Frame | Relative spatial distance |
| **74 .. 76** | `walker/ref_displacement[4]` | $[\Delta x_4, \Delta y_4, \Delta z_4]$ | Lookahead displacement to target CoM at $t + 4$ step ($+0.8\text{ ms}$) | $\text{cm}$ | Fly Egocentric Frame | Relative spatial distance |
| **77 .. 79** | `walker/ref_displacement[5]` | $[\Delta x_5, \Delta y_5, \Delta z_5]$ | Lookahead displacement to target CoM at $t + 5$ step ($+1.0\text{ ms}$) | $\text{cm}$ | Fly Egocentric Frame | Relative spatial distance |
| **80 .. 83** | `walker/ref_root_quat[0]` | $[w_0, x_0, y_0, z_0]$ | Relative orientation quaternion to target at current step ($t$) | Quaternion | Fly Egocentric Frame | Unit quaternion $\|\mathbf{q}\| = 1.0$ |
| **84 .. 87** | `walker/ref_root_quat[1]` | $[w_1, x_1, y_1, z_1]$ | Relative orientation quaternion to target at $t + 1$ step | Quaternion | Fly Egocentric Frame | Unit quaternion $\|\mathbf{q}\| = 1.0$ |
| **88 .. 91** | `walker/ref_root_quat[2]` | $[w_2, x_2, y_2, z_2]$ | Relative orientation quaternion to target at $t + 2$ step | Quaternion | Fly Egocentric Frame | Unit quaternion $\|\mathbf{q}\| = 1.0$ |
| **92 .. 95** | `walker/ref_root_quat[3]` | $[w_3, x_3, y_3, z_3]$ | Relative orientation quaternion to target at $t + 3$ step | Quaternion | Fly Egocentric Frame | Unit quaternion $\|\mathbf{q}\| = 1.0$ |
| **96 .. 99** | `walker/ref_root_quat[4]` | $[w_4, x_4, y_4, z_4]$ | Relative orientation quaternion to target at $t + 4$ step | Quaternion | Fly Egocentric Frame | Unit quaternion $\|\mathbf{q}\| = 1.0$ |
| **100 .. 103** | `walker/ref_root_quat[5]` | $[w_5, x_5, y_5, z_5]$ | Relative orientation quaternion to target at $t + 5$ step | Quaternion | Fly Egocentric Frame | Unit quaternion $\|\mathbf{q}\| = 1.0$ |

---

## 3. Breakdown of the 25 Active Joint Observables (`joints_pos` & `joints_vel`)
In flight mode (`disable_legs = True`), legs are retracted and removed from the active kinematic tree. The 25 active joints are ordered as follows:

1. `head_abduct`: Head lateral roll / abduction
2. `head_twist`: Head yaw / axial rotation
3. `head`: Head pitch / nodding
4. `wing_yaw_left`: Left wing stroke angle
5. `wing_roll_left`: Left wing elevation / deviation
6. `wing_pitch_left`: Left wing angle of attack / feathering
7. `wing_yaw_right`: Right wing stroke angle
8. `wing_roll_right`: Right wing elevation / deviation
9. `wing_pitch_right`: Right wing angle of attack / feathering
10. `abdomen_abduct`: Abdominal segment 1 lateral yaw
11. `abdomen`: Abdominal segment 1 flexion / pitch
12. `abdomen_abduct_2`: Abdominal segment 2 lateral yaw
13. `abdomen_2`: Abdominal segment 2 pitch
14. `abdomen_abduct_3`: Abdominal segment 3 lateral yaw
15. `abdomen_3`: Abdominal segment 3 pitch
16. `abdomen_abduct_4`: Abdominal segment 4 lateral yaw
17. `abdomen_4`: Abdominal segment 4 pitch
18. `abdomen_abduct_5`: Abdominal segment 5 lateral yaw
19. `abdomen_5`: Abdominal segment 5 pitch
20. `abdomen_abduct_6`: Abdominal segment 6 lateral yaw
21. `abdomen_6`: Abdominal segment 6 pitch
22. `abdomen_abduct_7`: Abdominal segment 7 lateral yaw
23. `abdomen_7`: Abdominal segment 7 pitch
24. `haltere_left`: Left gyroscopic balancing organ
25. `haltere_right`: Right gyroscopic balancing organ

---

## 4. Fundamental Finding: Absolute State vs Sensory Information

> **Question**: *Does the existing flight controller receive absolute state information or sensory information?*

### The Definitive Finding:
**The flight controller receives purely EGOCENTRIC SENSORY and RELATIVE TRACKING information, with ZERO absolute world state coordinates.**

### Proof from Source Code:
1. **No World Position**: The global Cartesian position $(X, Y, Z)_{world}$ from `physics.data.qpos[:3]` is **never passed** to the policy. The policy is completely blind to where it is located in world space.
2. **No World Orientation**: The world quaternion $\mathbf{q}_{world}$ is **never passed** directly. Instead, only two relative rotational quantities are provided:
   - `walker/world_zaxis`: The unit vertical gravity vector rotated into the fly's body frame (`observable.MJCFFeature('xmat', root)[6:]`). This mimics the insect's **vestibular / haltere graviceptive system**, giving tilt (pitch and roll) relative to gravity without knowing global yaw/compass heading.
   - `walker/ref_root_quat`: The relative orientation difference between the fly and the target in local coordinates (`flybody.quaternions.get_dquat_local(fly_quat, ref_quat)`).
3. **No World Velocities**: `velocimeter`, `accelerometer`, and `gyro` sensors strictly measure motion **in the thorax's instantaneous local reference frame**.
4. **Target Information is Egocentric**: Target tracking vectors in `walker/ref_displacement` are explicitly transformed into the fly's egocentric coordinates via:
   ```python
   self._walker.transform_vec_to_egocentric_frame(physics, ref_pos - fly_pos)
   ```

### Scientific Implication for Our Experiment:
When we design the custom interactive task (Milestone 3), the fly **does not need to know where it is in the room**. It only needs to observe:
* Its internal sensory state (tilt, velocities, joint angles).
* The **relative direction and distance vector** pointing toward the candidate target options (e.g., Target `[ 5 ]` at local offset $[+2.0, -1.0, +0.5]\text{ cm}$ vs Target `[ 7 ]` at local offset $[+2.0, +1.0, +0.5]\text{ cm}$).
This matches biological fly navigation, which relies on retinal optic flow and localized sensory cues rather than a global GPS.
