# FlyBody Architecture & System Reference

## 1. System Overview & Dependencies
The **FlyBody** simulation platform (Vaxenburg et al., *Nature* 2025; Google DeepMind & HHMI Janelia Research Campus) is an anatomically detailed physics model of the fruit fly (*Drosophila melanogaster*) implemented in [MuJoCo](https://github.com/google-deepmind/mujoco) and structured around DeepMind's `dm_control` and `dm_env` frameworks.

### Python & Engine Versions
* **Python Requirement**: Python $\ge 3.10$ (officially specified in `pyproject.toml`). Python 3.10 is specifically recommended to guarantee ABI stability with precompiled C-extensions, TensorFlow 2.8, and `dm_control`.
* **MuJoCo Version**: MuJoCo 3.1.3+ / 3.13.0.
* **NumPy Version**: `numpy==1.26.4` (strict requirement in `pyproject.toml` due to C-API compatibility).
* **Core Dependencies**: `dm_control`, `h5py`, `mediapy`, `pytest`, `scipy`, `imageio`, `torch` (CPU mode).

### Installation Procedure
```bash
# Ubuntu 26.04 / Linux (Conda environment)
conda create -n flybody -c conda-forge python=3.10 pip ipython
conda activate flybody
pip install numpy==1.26.4
pip install -e ./upstream/flybody
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Windows 11 / uv Virtual Environment (Active)
uv venv .venv --python 3.10
uv pip install numpy==1.26.4
uv pip install -e ./upstream/flybody
uv pip install -r requirements.txt
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 2. Anatomical Model Structure, Bodies & Degrees of Freedom
The fruit fly body model (`fruitfly.xml`) comprises **135 bodies** (including reference ghost frames) and **108 degrees of freedom** (103 joints):
* **Thorax**: Central rigid root body connected to the world via a 6-DoF free joint (`qpos[:3]` Cartesian position, `qpos[3:7]` unit quaternion orientation).
* **Head**: Articulated with 3 rotational DoFs (`head_abduct`, `head_twist`, `head`). Supports antennae and rostrum/haustellum proboscis mouthparts.
* **Abdomen**: 7-segment articulated chain (`abdomen_1` through `abdomen_7`), providing pitch and yaw stabilization in flight.
* **Wings**: Left and right aerodynamic blades with 3 rotational DoFs each:
  * `wing_yaw` (stroke angle / forward-backward oscillation)
  * `wing_roll` (stroke deviation / elevation)
  * `wing_pitch` (feathering / angle-of-attack)
* **Legs**: 6 articulated appendages (Foreleg T1, Midleg T2, Hindleg T3) with coxa, femur, tibia, and 4-segment tarsi ending in adhesive claws. In flight tasks, legs are retracted and disabled to optimize computation.
* **Halteres**: Bilateral gyroscopic sensing organs oscillating opposite to wing beats.

---

## 3. Actuators, Aerodynamics & Sensors
* **Wing Flapping & WBPG**: Flapping at 200–250 Hz is governed by the **Wing Beat Pattern Generator (WBPG)** (`flybody.tasks.pattern_generators.WingBeatPatternGenerator`). The neural controller does not command 20,000 Hz raw muscle twitches; instead, it outputs 12 kinematic bias parameters at 5,000 Hz (`control_dt = 0.0002 s`), which modulate baseline wing stroke trajectories.
* **Aerodynamic Fluid Model**: `flybody.ellipsoid_fluid_model.EllipsoidFluidModel` computes quasi-steady lift, drag, and rotational damping forces on ellipsoidal approximations of wings and body parts at each physics substep (`dt = 0.00005 s`, 20,000 Hz).
* **Sensors**:
  * `walker/accelerometer`: 3D specific force at thorax CoM.
  * `walker/gyro`: 3D angular velocity vector ($\text{rad/s}$).
  * `walker/joints_pos` & `joints_vel`: Proprioceptive encoders across active joints.
  * `walker/touch`: Force sensors on tarsi and antennae.
  * Virtual compound eyes: Retina array generating optic flow and luminance patterns (`vision_guided_flight`).

---

## 4. Available Environments in FlyBody
All environments are constructed in `flybody.fly_envs`:
1. `flight_imitation(ref_path=None, ...)`:
   * **Task**: Track real fruit-fly flight saccades or arbitrary spatial trajectories.
   * **Action Space**: `(12,)` continuous controls in $[-0.2, 0.2]$ (or $[-3.0, 3.0]$ depending on scaling).
   * **Observation Space**: `(104,)` continuous features (velocities, quats, lookaheads).
   * **Frequency**: Physics at 20,000 Hz, control at 5,000 Hz.
2. `walk_imitation(ref_path=None, ...)`:
   * **Task**: Ground locomotion tracking over flat floors.
   * **Action Space**: `(59,)` joint position/force actuators.
   * **Observation Space**: `(741,)` joint and appendage positions, forces, and gyros.
   * **Frequency**: Physics at 5,000 Hz, control at 500 Hz.
3. `walk_on_ball(...)`: Locomotion on a suspended spherical floating ball.
4. `vision_guided_flight(bumps_or_trench='bumps')`: Flight navigation through 3D sinusoidal terrain using compound eye visual inputs (`(6208,)` observation space).
5. `template_task(...)`: Lightweight template environment for constructing custom RL tasks.

---

## 5. Machine Learning & Training Frameworks
* **Existing Reinforcement Learning**: DeepMind DMPO (Distributed Distributional MPO) implementation in `flybody.agents`, utilizing Ray for multi-worker parallel rollouts and TensorFlow/Acme for learner networks (`train_dmpo_ray.py`).
* **Existing Imitation Learning**: Supervised behavior cloning across high-speed kinematic flight data (recorded at 5 kHz by HHMI Janelia).
* **Datasets & Checkpoints (Figshare DOI: 10.25378/janelia.25309105)**:
  * `flight-imitation-dataset` (saccade evasions)
  * `walking-imitation-dataset` (markerless walking poses)
  * `trained-policies` (DMPO policies for walking and flight)
  * Download utility provided in `flybody.download_data.figshare_download`.

---

## 6. Simulation State Access & Control API
* **Environment Instantiation**:
  ```python
  from flybody.fly_envs import flight_imitation
  env = flight_imitation()
  time_step = env.reset()
  ```
* **Reading Observations**:
  ```python
  obs_dict = time_step.observation  # dict of named sensor arrays
  reward = time_step.reward
  is_terminal = time_step.last()
  ```
* **Accessing Raw MuJoCo Physics State**:
  ```python
  physics = env.physics
  pos = physics.data.qpos[:3]       # Thorax Cartesian CoM (cm)
  quat = physics.data.qpos[3:7]     # Orientation quaternion [w, x, y, z]
  vel = physics.data.qvel[:3]       # Linear velocity (cm/s)
  omega = physics.data.qvel[3:6]    # Angular velocity (rad/s)
  ```
* **Sending Actions to Fly**:
  ```python
  import numpy as np
  action = np.zeros(env.action_spec().shape, dtype=np.float32)
  time_step = env.step(action)
  ```
* **Offscreen Camera Rendering**:
  ```python
  frame = env.physics.render(height=480, width=640, camera_id=0)  # camera 0: 'hero'/'track'
  ```
