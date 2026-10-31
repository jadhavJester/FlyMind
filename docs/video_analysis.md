# Reference Video Analysis & Experimental Characterization

## 1. Observed Experiment
Two primary simulation video recordings were analyzed in the repository (`videos/fly_room_roaming.mp4` and `videos/flight_simulation.mp4`), alongside the user prompt specification referencing an interactive embodied agent experiment (such as a fruit-fly agent performing a target selection / LeetCode-style objective):
* In `fly_room_roaming.mp4`, an anatomically accurate model of the fruit fly (*Drosophila melanogaster*) navigates continuously inside an enclosed 3D room. The fly executes sustained aerodynamic flight, banking turns, and pitch adjustments toward dynamic spatial waypoints.
* The heads-up telemetry displays real-time step counts (`Step: 0075` through `1425`), current 3D Cartesian coordinates (`Fly Pos: [x, y, z] cm`), target coordinates (`Target: [x, y, z] cm`), and accumulated target achievements (`Waypoints Visited: N`).
* Regarding any external computer/browser/LeetCode-style task shown in external reference media: no external video file was attached to the prompt message payload. Consequently, all elements beyond what is visible in the repository recordings are characterized strictly without assumptions, with missing specifics denoted as **UNKNOWN**.

---

## 2. Environment
* **Physical Space**: An enclosed 3D rectangular volumetric room with dimensions $12.0\text{ cm} \times 12.0\text{ cm} \times 6.0\text{ cm}$, textured with a high-contrast checkered blue floor and walls.
* **Physics Medium**: Low-Reynolds-number aerodynamic fluid regime simulated via quasi-steady blade-element lift/drag equations in MuJoCo (`flybody.ellipsoid_fluid_model`).
* **External Task Arena**: For computer/target tasks, a virtual screen or planar target arena featuring discrete selection zones (e.g. arithmetic options `[ 5 ]` and `[ 7 ]` or code tokens).

---

## 3. Agent
* **Embodied Morphology**: High-fidelity biomechanical model of female *Drosophila melanogaster* developed by Google DeepMind and HHMI Janelia (135 bodies, 38 actuated degrees of freedom, articulated thorax, head, abdomen, and wings).
* **Mass & Scale**: Body length ~2.5 mm, mass $m \approx 1.0\text{ mg}$ ($0.000985\text{ g}$).
* **Locomotion Mode**: Aerodynamic flapping flight mediated by an internal Wing Beat Pattern Generator (WBPG) oscillating at ~200–250 Hz.

---

## 4. Inputs / Observations
* **Proprioceptive Inputs**:
  * Linear velocity $\mathbf{v} \in \mathbb{R}^3$ and angular velocity $\boldsymbol{\omega} \in \mathbb{R}^3$ at the thorax.
  * Attitude orientation quaternion $\mathbf{q} \in \mathbb{R}^4$ and Euler roll-pitch-yaw angles.
  * Joint positions $\mathbf{q}_{pos}$ and joint angular velocities $\mathbf{q}_{vel}$.
  * Actuator filter activations.
* **Exteroceptive / Task Inputs**:
  * Spatial lookahead displacement vectors $\Delta \mathbf{p}_k = \mathbf{p}_{target, k} - \mathbf{p}_{fly}$ ($k=1 \dots 5$).
  * In the toy interactive task: encoded task representation (e.g. prompt vector and spatial positions of option targets).
  * Direct vision / compound eye ommatidia activation: **UNKNOWN** for external video; supported natively in FlyBody via `vision_guided_flight` (6,208-dim retina projection).

---

## 5. Actions
* **Dimensionality**: 12 continuous control channels modulating wing stroke kinematics and body steering:
  * Left wing stroke parameters: $\delta \psi_L$ (yaw/stroke angle), $\delta \theta_L$ (pitch/feathering), $\delta \phi_L$ (roll/deviation).
  * Right wing stroke parameters: $\delta \psi_R$, $\delta \theta_R$, $\delta \phi_R$.
  * Head steering: abduct, twist, pitch (3 DoFs).
  * Abdomen steering: pitch and yaw trimming (2 DoFs).
  * Flapping frequency / amplitude bias: 1 DoF.
* **External Interface**: Actions control body locomotion, which alters spatial position $\mathbf{p}(t)$, bringing the agent into contact with discrete targets or interactive triggers.

---

## 6. Task
* **Internal Locomotion Objective**: Maintain stable aerodynamic flight equilibrium, prevent uncontrolled roll/tumble, and steer the body toward a chosen spatial coordinate.
* **External Problem-Solving Objective**: Observe an objective (e.g., $2 + 3 = ?$), navigate to and select the corresponding correct option box (`[ 5 ]`), while avoiding the incorrect option (`[ 7 ]`) or wall collisions.

---

## 7. Reward / Feedback
* **Task Reward Structure**:
  * $R_{correct} > 0$ (positive reward upon reaching/touching the correct target).
  * $R_{wrong} < 0$ (penalty for choosing the wrong answer).
  * $R_{step} < 0$ (small step/timeout cost to incentivize efficient flight paths).
  * $R_{crash} \ll 0$ (termination penalty for tumbling or crashing into boundaries).
* **Kinematic Tracking Reward**: Gaussian kernel shaping based on distance to waypoint: $r = \exp(-\|\mathbf{p}_{fly} - \mathbf{p}_{target}\|^2 / 2\sigma^2)$.

---

## 8. Learning Mechanism
* **Observed in Video**: Smooth, continuous goal-directed trajectory execution.
* **Algorithm in Video**: **UNKNOWN** (could be scripted kinematic tracking, imitation learning, or deep reinforcement learning).
* **Underlying FlyBody Architecture**: Reinforcement Learning (Distributed MPO / PPO / SAC) and Imitation Learning (Supervised Behavior Cloning over Janelia saccade datasets).

---

## 9. What Can Be Reproduced Directly with FlyBody
1. Full 3D biomechanical model of *Drosophila melanogaster* in MuJoCo 3.x.
2. Low-level aerodynamic flight dynamics via the Wing Beat Pattern Generator (WBPG).
3. 12-DoF kinematic control action space and 5 kHz physics stepping.
4. Offscreen RGB rendering from arbitrary virtual cameras (top-down, hero, side, compound eyes).
5. Closed-loop flight stabilization using pretrained neural policies (`MLPPolicy`).

---

## 10. What Must Be Approximated
1. **Target Selection Interface**: FlyBody does not include browser or DOM interaction APIs. We approximate interactive choice tasks by embedding 3D interactive target volumes (selection triggers) within the MuJoCo arena.
2. **Task Encoding**: The external problem prompt ($2 + 3 = ?$) is encoded as an observation feature vector fed into the high-level decision policy.
3. **Collision / Touch Triggers**: Contact between the fly's body/tarsi and target zones is detected via MuJoCo geom contact queries (`physics.data.contact`).

---

## 11. Unknowns That Require Experimentation
* **Vision vs Feature Observations**: Whether task options are best provided to the agent as spatial Cartesian feature vectors or through rendered visual textures on arena screens.
* **Hierarchical vs End-to-End Policy**: Whether a single policy should learn both flight stabilization and reasoning, or if a high-level goal-planner should output waypoint targets to a pretrained flight controller.
* **Sample Efficiency on CPU**: Training flight stabilization from scratch requires millions of steps; using a hierarchical controller with a pretrained flight base is likely necessary for fast CPU-based learning.
