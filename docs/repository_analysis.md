# FlyMind: Comprehensive Repository Investigation & Technical Analysis

This document provides a source-code-level inspection of seven primary open-source insect-neuroscience, connectome-robotic, and neuromorphic-vision projects. It evaluates their software architectures, core mathematical algorithms, input/output interfaces, licensing terms, and technical reusability for the **FlyMind** autonomous civilian UAV research platform.

---

## 1. Master Repository Comparison & Reusability Matrix

| Repository | Primary Architecture & Language | Core Algorithms | Licensing | Reusable Modules for FlyMind | Non-Reusable / Discarded Elements |
|---|---|---|---|---|---|
| **A. FlyBody**<br>*(Turaga Lab, Nature 2025)* | Python 3.10+, C++<br>MuJoCo 3.13, dm_control | Quasi-steady fluid dynamics (`EllipsoidFluidModel`), WBPG periodic table, egocentric vector transforms | Apache 2.0 | • `transform_vec_to_egocentric_frame`<br>• Linear/quadratic bluff-body aerodynamic drag<br>• Forward lookahead trajectory interpolation | • Flapping Wing Beat Pattern Generator (WBPG)<br>• 135-body / 38-joint biomechanical fly skeleton |
| **B. FlyBrain**<br>*(Dylan Kainth)* | Python 3.10+, OpenCV<br>PyTorch, TelloPy | Retinotopic hexagonal eye-map projection, evolutionary weight search, PID telemetry | MIT | • 2D camera-to-ommatidia projection mapping (`flybrain_eye_map.py`)<br>• Real-time OpenCV neural telemetry visualizer | • Unconstrained genetic algorithm optimization<br>• Hardware-coupled DJI Tello flight commands |
| **C. FlyDrones**<br>*(SpikeCalls, 2026)* | Python 3.10+, NumPy, SciPy<br>Three.js WebGL (Browser) | Leaky Integrate-and-Fire (LIF) network, Lucas-Kanade 4-channel optic flow, looming expansion, haltere damping | MIT | • Vectorized event-driven LIF simulation engine (`lif.py`)<br>• Motor decoder & Safety Governor (`safety.py`)<br>• Interactive 3D WebGL research dashboard (`docs/live/`) | • Hard-coded MiniFly synaptic weights<br>• Heavy whole-brain graph traversal for real-time UAV loops |
| **D. Fly.ai**<br>*(Alex Titonis)* | Python 3.10+, PyTorch<br>Crazyflie simulation | Connectome reservoir computing, visual imprinting, decoupled reflex vs. planner architecture | MIT | • Separation of sub-10ms *reflex* (collision avoidance) vs. *planner* (heading guidance)<br>• Crazyflie simulation wrapper | • Social/Discord bot automation modules<br>• Non-standardized neural reward schemas |
| **E. EMD-LPLC2-GF**<br>*(Junyu Zhao et al., iScience 2023)* | MATLAB / Octave<br>(Reproducible in PyTorch/NumPy) | 2D Hassenstein-Reichardt EMD, 4-lobed cross-kernel multiplicative LPLC2 integration, LIF Giant Fiber | MIT / Academic Open | • Mathematical 4-channel EMD filter formulation<br>• LPLC2 cross-kernel receptive field convolution<br>• Solid-angle invariant threshold trigger ($\theta \approx 34^\circ$) | • Offline MATLAB batch scripts without real-time UAV control |
| **F. Speck Optical Flow**<br>*(TU Delft MAVLab, ICRA 2026)* | Python 3.10+, C++/CUDA<br>PyTorch, ROS 2 Humble | Event-based optical flow via hybrid SNN-ANN (minGRU recurrent core + conv decoder), contrast maximization | MIT | • Neuromorphic event-flow contrast maximization equations<br>• ROS 2 PX4 offboard velocity controller structure (`speckflow_control`) | • Proprietary SynSense Speck2e devkit firmware binaries (`motherBoardV2_0_11_5.img`) |
| **G. EVFly**<br>*(Bhattacharya et al., CoRL 2024)* | Python 3.8/3.10, C++<br>ROS Noetic / ROS 2, PyTorch | Monocular event-based depth pretext task, reactive obstacle avoidance policy, Flightmare/Unity sim | MIT | • Pretext depth/nearness learning framework<br>• Cluttered obstacle forest world configurations (100 procedural tree maps)<br>• Metric benchmark evaluation harness | • Heavy Flightmare Unity binary dependencies<br>• Python 3.8 legacy Catkin workspace coupling |

---

## 2. Detailed Code-Level Investigation of Each Repository

### A. FlyBody (TURAG Lab / Google DeepMind / HHMI Janelia)
* **Location in Workspace**: [`upstream/flybody/`](file:///d:/Projects/Fruit%20fly/upstream/flybody)
* **Core Source Files**:
  - `flybody/fruitfly/fruitfly.py`: Top-level biomechanical fly entity, camera placements, body-part groupings.
  - `flybody/tasks/pattern_generators.py`: Central Pattern Generator (CPG) maintaining discrete stroke tables for 218 Hz wing flapping.
  - `flybody/tasks/flight_imitation.py`: Trajectory generation, reference ghost fly positioning, and tracking observation extraction.
* **Algorithmic Inspection**:
  - FlyBody models the fluid dynamics of insect flight using an `EllipsoidFluidModel` attached to the flapping wings. It computes translational lift, drag, and rotational damping torques based on the instantaneous velocity of each wing segment relative to the surrounding fluid:
    $$\mathbf{F}_{\text{aero}} = \frac{1}{2} \rho A v^2 \left( C_L(\alpha) \hat{\mathbf{n}}_L + C_D(\alpha) \hat{\mathbf{n}}_D \right)$$
* **Reusability Evaluation**:
  - **Reusable**: The mathematical formulation of aerodynamic drag and the coordinate transformation utilities (`transform_vec_to_egocentric_frame`).
  - **Not Reusable for Quadrotors**: FlyBody’s actuators model cyclic wing stroke plane kinematics. Quadrotors use independent brushless DC motor thrusts, governed by Newton-Euler 6-DoF rigid-body mechanics.

---

### B. FlyBrain (Dylan Kainth)
* **Location in Workspace**: [`upstream/flybrain/`](file:///d:/Projects/Fruit%20fly/upstream/flybrain)
* **Core Source Files**:
  - `flybrain_eye_map.py`: Projects rectangular camera pixels onto a biological hexagonal ommatidial lattice matching Drosophila compound eyes.
  - `drone_brain_controller.py`: Ingests simulated/hardware sensory vectors and maps descending neurons to PID setpoints.
  - `visualizer_realtime.py`: High-speed OpenCV canvas displaying neural activations, eye maps, and flight metrics.
* **Algorithmic Inspection**:
  - Maps Cartesian coordinates $(u, v)$ from a camera image into polar/hexagonal visual receptive fields $(r, \phi)$, mirroring the ~750 ommatidia per eye in fruit flies.
* **Reusability Evaluation**:
  - **Reusable**: The eye-mapping projection script provides a fast, vectorized method to convert raw camera frames into biologically clustered visual signals.
  - **Limitations**: The evolutionary training script (`phase11c_evolutionary_improved.py`) is unconstrained and prone to overfitting on narrow flight corridors.

---

### C. FlyDrones (SpikeCalls)
* **Location in Workspace**: [`upstream/FlyDrones/`](file:///d:/Projects/Fruit%20fly/upstream/FlyDrones)
* **Core Source Files**:
  - `src/flydrones/brain/lif.py`: Event-driven Leaky Integrate-and-Fire simulator with refractory periods and decaying synaptic conductances.
  - `src/flydrones/senses/retina.py`: Computes 4-channel directional optical flow (`ftb`, `btf`, `up`, `down`) and optical expansion rates (`loom`).
  - `src/flydrones/motor/decoder.py`: Decodes firing rates from `DNg02` (wing stroke amplitude) into linear throttle and yaw rate.
  - `src/flydrones/safety.py`: Independent safety governor clamping acceleration slew rates, ceiling/floor limits, and watchdog timeouts.
  - `docs/live/`: Standalone WebGL/Three.js interactive 3D simulation running the spiking brain loop live in browser.
* **Algorithmic Inspection**:
  - Solves the differential membrane potential equation:
    $$\tau_m \frac{dV_i}{dt} = -(V_i - V_{\text{rest}}) + R_m \sum_j W_{ij} S_j(t)$$
    When $V_i \ge V_{\text{thresh}}$, neuron $i$ spikes, resetting to $V_{\text{reset}}$ for refractory duration $t_{\text{ref}}$.
* **Reusability Evaluation**:
  - **Directly Reusable**: The safety governor architecture (`SafetyGovernor`), the 4-channel retinal flow preprocessor, and the interactive WebGL dashboard engine.

---

### D. Fly.ai (Alex Titonis)
* **Location in Workspace**: [`upstream/fly_ai/drone/`](file:///d:/Projects/Fruit%20fly/upstream/fly_ai/drone)
* **Core Source Files**:
  - `reflex.py`: Sub-10ms fast reactive evasion layer for emergency obstacles.
  - `planner.py`: High-level goal-directed waypoint generator.
  - `pilot.py`: Blends long-horizon planner objectives with short-horizon reflex interventions.
  - `sim/`: Physics wrapper around Bitcraze Crazyflie quadrotor dynamics.
* **Algorithmic Inspection**:
  - Implements a dual-rate control architecture: high-frequency reflexive avoidance overrides low-frequency path planning when visual looming metrics exceed safety margins.
* **Reusability Evaluation**:
  - **Reusable**: The architectural pattern separating *reflex* from *planning*. This directly satisfies FlyMind’s Module 4 (Safe Navigation Decision System).

---

### E. EMD-LPLC2-GF Model (Junyu Zhao et al.)
* **Location in Workspace**: [`upstream/EMD-LPLC2-GF-model/`](file:///d:/Projects/Fruit%20fly/upstream/EMD-LPLC2-GF-model)
* **Core Source Files**:
  - `code/1_Response-Characteristics-.../emd.m`: Two-armed Hassenstein-Reichardt Elementary Motion Detector model.
  - `code/1_Response-Characteristics-.../lplc2conv2.m`: Cross-shaped 4-lobed receptive field convolution kernel.
  - `code/1_Response-Characteristics-.../LIF.m`: Leaky Integrate-and-Fire Giant Fiber model.
  - `code/6_Robot-Simulation-About-Goal-Directed-Tasks/`: 2D differential robot navigation in cylindrical obstacle arenas.
* **Algorithmic Inspection**:
  - Multiplicative integration across 4 cardinal arms of each LPLC2 receptive field:
    $$\text{LPLC2}_{i,j}(t) = \left[ L_{\text{up}} - L_0 \right]_+ \cdot \left[ L_{\text{down}} - L_0 \right]_+ \cdot \left[ L_{\text{left}} - L_0 \right]_+ \cdot \left[ L_{\text{right}} - L_0 \right]_+$$
    This mathematical structure guarantees zero response to laminar translation and maximal response to isotropic radial expansion (looming).
* **Reusability Evaluation**:
  - **Directly Reproducible in Python**: The mathematical equations translate cleanly into vectorized PyTorch/NumPy tensor convolutions.

---

### F. Speck Optical Flow (TU Delft MAVLab, ICRA 2026)
* **Location in Workspace**: [`upstream/speck-optical-flow/`](file:///d:/Projects/Fruit%20fly/upstream/speck-optical-flow)
* **Core Source Files**:
  - `speckflow_ros2/speckflow_control/`: ROS 2 Humble node providing closed-loop optical flow flight control for PX4.
  - `training/`: Self-supervised contrast maximization framework for dense event-based optical flow.
  - `speck_inference/`: Neuromorphic SNN-ANN hybrid deployment pipeline.
* **Algorithmic Inspection**:
  - Warps spatio-temporal visual events by candidate velocity vectors $(\hat{u}, \hat{v})$ and maximizes image sharpness (variance of the event count image):
    $$\mathcal{L}_{\text{cmax}}(\hat{\mathbf{v}}) = -\text{Var}\left( I_{\text{warped}}(\mathbf{x}; \hat{\mathbf{v}}) \right)$$
* **Reusability Evaluation**:
  - **Reusable**: The ROS 2 PX4 offboard velocity controller and state machine architecture.
  - **Hardware Dependency**: The Speck SNN inference scripts require proprietary SynSense devkit hardware; FlyMind will reproduce the algorithm via software emulation.

---

### G. EVFly (Bhattacharya et al., GRASP Lab / UZH RPG, CoRL 2024)
* **Location in Workspace**: [`upstream/evfly/`](file:///d:/Projects/Fruit%20fly/upstream/evfly)
* **Core Source Files**:
  - `learner/`: Deep neural network predicting monocular depth as a pretext task for reactive avoidance.
  - `flightmare/`: Simulation bindings connecting quadrotor dynamics to Unity rendering.
  - `dodgedrone_simulation/`: Benchmark environments evaluating agile flight through dynamic obstacle fields.
* **Algorithmic Inspection**:
  - Trains a reactive policy mapping visual streams directly to body-rate setpoints $(\omega_x, \omega_y, \omega_z, a_z)$, showing that higher speeds ($>3\text{ m/s}$) produce crisper motion cues for collision estimation than stationary hovering.
* **Reusability Evaluation**:
  - **Reusable**: Benchmark methodology, evaluation metric definitions (clearance distance, time-to-collision, collision frequency), and procedural obstacle configurations.
  - **Limitations**: Unity binary dependencies (`flightrender.tar`) are heavy (443 MB) and require specific GPU setups. FlyMind will implement a lighter, native 6-DoF simulator while maintaining EVFly’s benchmark rigor.

---

## 3. FlyMind Reproduction & Implementation Strategy

Based on the code inspection, FlyMind will synthesize these research assets into a unified architecture:

```
[Camera & IMU] ──► [Bio-Inspired Retina (FlyDrones + FlyBrain)]
                         │
                         ├──► [2D EMD & LPLC2 Looming (Zhao et al.)] ──► [Giant Fiber Reflex]
                         │                                                       │
                         ├──► [Nearness Map & Divergence (Paper 4)]              │
                         │                                                       │
                         └──► [WTA Gap Finding (Schoepe et al.)] ──► [Heading Selector]
                                                                          │
                                                                   [Safety Supervisor]
                                                                          │
                                                                   [PX4 / 6-DoF UAV]
```

1. **Perception**: Combine **FlyDrones'** 4-channel retinal optic flow with **Zhao et al.'s** 4-lobed LPLC2 cross-kernel convolution in vectorized PyTorch.
2. **Looming Detection**: Implement the invariant angular-size threshold ($\theta \approx 34^\circ$) from **Zhao et al.** and IMU ego-motion compensation from **Paper 3**.
3. **Gap Selection**: Reproduce the neuromorphic Winner-Take-All lateral inhibition from **Schoepe et al. (Nature Comm 2024)** to identify obstacle-free flight corridors.
4. **Safety & Control**: Adopt **FlyDrones'** independent safety supervisor pattern and **Fly.ai's** reflex/planner dual-rate hierarchy.
5. **Evaluation Harness**: Adopt **EVFly's** standardized benchmark protocol (collision rate, minimum clearance, navigation completion, latency).
