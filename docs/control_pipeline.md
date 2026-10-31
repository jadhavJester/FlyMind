# FlyBody Control Pipeline & Architectural Hierarchy

## 1. The Three-Level Control Architecture

FlyBody organizes biological flight control into three strictly delineated functional tiers:

```mermaid
flowchart TD
    subgraph Level 3: High-Level / Learned Policy (5,000 Hz)
        A[Task Objective / Target Coordinates] --> B[Observation Vector: 104 dims\nVestibular, Proprioception, Target Displacement]
        B --> C[Neural Policy / Agent Network]
        C --> D[12 Action Parameters\nHead 3, Wings 6, Abdomen 2, Frequency 1]
    end

    subgraph Level 2: Low-Level Biological Controller (WBPG & Actuators)
        D --> E[Wing Beat Pattern Generator\nWBPG @ 218 Hz +/- 10%]
        E --> F[Cyclic Wing Kinematics\nYaw, Roll, Pitch @ 5,000 Hz]
        D -.-> G[Head & Abdomen PD Actuators]
        F --> H[Combined Wing Target Kinematics\nq_desired = q_wbpg + a_wing]
        H --> I[Virtual Spring Force Conversion\nu_wings = a_wing + q_wbpg - q_actual]
    end

    subgraph Level 1: FlyBody Physics & Aerodynamics (20,000 Hz, 4 Substeps)
        I --> J[MuJoCo Physics Integrator\n135 Bodies, 38 Active Joints]
        G --> J
        J --> K[Ellipsoid Quasi-Steady Aerodynamics\nBlade Element Lift, Drag, Damping]
        K --> L[Aerodynamic Forces & Torques on Wings/Body]
        L --> J
        J --> M[Next Physical State: qpos, qvel, accelerations]
        M --> B
    end
```

---

## 2. Distinction Between the Three Levels

### Level 1: FlyBody Physics Engine (MuJoCo 3.x)
* **Frequency**: $20,000\text{ Hz}$ ($dt = 0.00005\text{ s}$).
* **Scope**: Completely deterministic physical simulation.
* **Components**:
  * Newton-Euler equations of motion across 135 bodies and 38 active joints.
  * Passive structural dynamics: joint stiffness ($k = 0.05$), damping ($c = 0.001$), joint limits, spring references.
  * `EllipsoidFluidModel`: Quasi-steady fluid dynamic interaction computing instantaneous angle of attack $\alpha$, translational lift/drag, and rotational damping torques $\boldsymbol{\tau}_{damp}$ on wing ellipsoids.
* **Role**: The physical environment where forces create biological movement.

### Level 2: Low-Level Biological Controller (Wing Beat Pattern Generator)
* **Frequency**: Oscillates at natural flapping frequency ($\approx 218\text{ Hz}$, $T \approx 4.6\text{ ms}$), evaluated at $5,000\text{ Hz}$.
* **Scope**: Hardcoded, neuro-mechanical Central Pattern Generator (CPG).
* **Components**:
  * `WingBeatPatternGenerator` (`flybody/tasks/pattern_generators.py`): Contains a biological baseline stroke pattern recorded from real *Drosophila* wing kinematics.
  * Seamless phase-preserving frequency switching across discrete beat tables.
  * Kinematic-to-force conversion converting position targets into joint torques.
* **Role**: Shields the agent from having to learn cyclic flapping oscillations from scratch.

### Level 3: Learned / High-Level Controller (Decision Agent)
* **Frequency**: $5,000\text{ Hz}$ ($\Delta t = 0.0002\text{ s}$).
* **Scope**: Where machine learning, reinforcement learning, and goal-directed behavior reside.
* **Components**:
  * Observation ingestion (104 sensory features).
  * Policy network (e.g. `MLPPolicy` or DMPO actor).
  * 12-dimensional output vector modulating stroke biases, body attitudes, and throttle.
* **Role**: Steers the insect to accomplish high-level navigation, evasion, and target selection.

---

## 3. What the Agent Can Actually Control

Based on source inspection and our empirical perturbation study (`results/milestone_02/action_perturbation/`):

1. **Wing Flapping Frequency (Throttle / Lift Gain)**:
   * **Action Channel**: `action[11]` (`wbpg_frequency_bias`).
   * **Control Range**: $[-1.0, +1.0] \to [196.2\text{ Hz}, 239.8\text{ Hz}]$.
   * **Physical Effect**: Increases or decreases global aerodynamic lift across both wings. Positive values induce immediate vertical climb and accelerate forward flight.
2. **Bank / Roll Torque (Differential Wing Roll)**:
   * **Action Channels**: `action[4]` (`wing_roll_left`) and `action[7]` (`wing_roll_right`).
   * **Physical Effect**: Asymmetric wing deviation tilts the stroke plane of one wing relative to the other. In our perturbation test, applying $+0.5\text{ rad}$ to the left wing and $-0.5\text{ rad}$ to the right wing generated strong lateral roll ($> 60^\circ$ bank within $30\text{ ms}$) and rapid coordinated turning.
3. **Pitch Attitude & Forward Thrust (Symmetric Wing Pitch)**:
   * **Action Channels**: `action[5]` (`wing_pitch_left`) and `action[8]` (`wing_pitch_right`).
   * **Physical Effect**: Alters the angle of attack $\alpha$ of the wings during forward and backward strokes, shifting the net aerodynamic force vector forward or backward.
4. **Body Trim & Yaw Steering (Abdomen Rudder)**:
   * **Action Channels**: `action[9]` (`abdomen_abduct`) and `action[10]` (`abdomen`).
   * **Physical Effect**: Abdominal deflection acts as a biological rudder. Lateral abduction (`abdomen_abduct`) creates yaw torque; flexion/extension (`abdomen`) shifts the center of mass, trimming pitch equilibrium.
5. **Gaze & Visual Stabilization (Head Joint Targets)**:
   * **Action Channels**: `action[0..2]` (`head_abduct`, `head_twist`, `head`).
   * **Physical Effect**: Re-orients the head in 3 DoFs, allowing gaze stabilization independent of body roll during high-speed saccadic maneuvers.

---

## 4. How Targeted Flight Works (The Existing Reference Trajectory Mechanism)

In `flight_imitation.py`:
1. **Trajectory Loader**: `self._traj_generator` loads 3D CoM positions and quaternions from HDF5 saccade recordings.
2. **Ghost Reference Fly**: A virtual, semi-transparent ghost fly (`self._ghost`) is spawned in the arena at target coordinates.
3. **Observation Delivery**:
   * At each step $t$, the task extracts the target position $\mathbf{p}_{ref}(t)$ and 5 lookahead points $\mathbf{p}_{ref}(t+k)$.
   * It computes the relative vector $\Delta \mathbf{p} = \mathbf{p}_{ref} - \mathbf{p}_{fly}$ and rotates it into the **fly's egocentric reference frame**:
     ```python
     self._walker.transform_vec_to_egocentric_frame(physics, ref_pos - fly_pos)
     ```
   * It computes relative orientation difference using local quaternion math (`get_dquat_local`).
4. **Reward Shaping**:
   * Displacement reward: $r_{disp} = \text{rewards.tolerance}(\|\mathbf{p}_{target} - \mathbf{p}_{fly}\|, \text{margin}=0.4\text{ cm})$.
   * Attitude alignment reward: $r_{quat} = \text{rewards.tolerance}(\text{quat\_dist}, \text{margin}=\pi)$.
   * Scalar step reward: $R_t = r_{disp} \times r_{quat}$.

---

## 5. Minimum Interface Required for Custom Tasks (The Adapter Architecture)

To insert our custom interactive decision task without modifying the underlying FlyBody source code, we use a clean **Gym/Gymnasium-style Adapter Wrapper**:

```python
class InteractiveTaskWrapper:
    """Clean adapter interfacing custom tasks to FlyBody flight dynamics."""
    
    def __init__(self, task_objective):
        self._env = flybody.fly_envs.flight_imitation(ref_path=None)
        self._task = task_objective

    def reset(self, seed=None):
        # 1. Reset underlying FlyBody simulation
        time_step = self._env.reset()
        # 2. Reset custom task state (e.g. prompt '2 + 3 = ?', spawn targets [5] and [7])
        custom_obs = self._task.get_task_observation(self._env.physics)
        return custom_obs, {}

    def step(self, action):
        # 1. Step low-level FlyBody physics with 12-channel action
        time_step = self._env.step(action)
        # 2. Evaluate task criteria (target contact, wrong answer, timeout)
        reward, terminated, truncated, info = self._task.evaluate_step(self._env.physics)
        # 3. Formulate next observation
        obs = self._task.get_task_observation(self._env.physics)
        return obs, reward, terminated, truncated, info
```

This architecture decouples task logic from flight physics, ensuring zero corruption of upstream FlyBody code.
