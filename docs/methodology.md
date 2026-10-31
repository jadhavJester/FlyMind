# FlyMind: Scientific Methodology & Theoretical Foundations

This document establishes the scientific research methodology for the **FlyMind** autonomous civilian UAV intelligence platform. It synthesizes insect visual neuroscience, computational motion vision, and neuromorphic engineering principles into a repeatable, testable framework.

---

## 1. Research Scope & Ethical Boundaries

FlyMind is strictly restricted to **civilian autonomous flight**:
* **Permitted Capabilities**: Environmental awareness, obstacle avoidance, corridor and gap navigation, emergency braking climbs, proximity heatmapping, and sensor-degraded safe landing.
* **Prohibited Capabilities**: Weaponization, target interception, offensive tracking, attack guidance, payload delivery, and evasion of defensive systems.

---

## 2. Core Scientific Hypotheses

1. **H1 (Multiplicative Gating for Looming Specificity)**:
   A 4-lobed cardinal cross-kernel convolution with multiplicative integration across opposing directional channels:
   $$\text{LPLC2}_{i,j}(t) = \prod_{k=1}^4 \left[ L_{i,j}^k(t) - L_0 \right]_+$$
   will completely suppress uniform laminar optic flow (background drift during translation) while producing selective, robust activation for isotropic visual expansion (looming obstacles).

2. **H2 (Invariant Angular Size Thresholding)**:
   Integration of the active LPLC2 population count $N_{\text{act}}(t)$ and its temporal derivative $\left[\frac{d N_{\text{act}}}{dt}\right]_+$ by a downstream Leaky Integrate-and-Fire Giant Fiber (GF) neuron triggers an escape command at an invariant subtended visual angle:
   $$\theta_{\text{trigger}} \approx 34.0^\circ$$
   irrespective of the obstacle's physical diameter $L$ or approach velocity $v$.

3. **H3 (Neuromorphic Gap Finding via Lateral Flow Minimization)**:
   Partitioning the forward visual field into $M = 16$ angular sectors and applying Winner-Take-All (WTA) lateral inhibition isolates the optic flow minimum ($m^* = \arg\min_m H_m$), directing the vehicle toward the deepest navigable opening without requiring 3D metric scene reconstruction.

4. **H4 (Ego-Motion Compensation via IMU Derotation)**:
   Subtracting gyroscope rotational optic flow $\mathbf{u}_{\text{rot}} = \mathbf{B}(\mathbf{x}) \boldsymbol{\omega}_{\text{IMU}}$ from the total flow field isolates true translational expansion, eliminating false looming triggers during aggressive quadrotor banking turns.

5. **H5 (Continuous Proximity Heatmapping via Flow Divergence)**:
   The 2D spatial divergence of the derotated flow field:
   $$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 2 \mu(\mathbf{x}) v_z$$
   provides a reliable metric-free proxy for relative nearness $\mu = \frac{1}{d}$ and instantaneous Time-to-Contact ($\text{TTC} = \frac{1}{\nabla \cdot \mathbf{u}}$).

---

## 3. Experimental Architecture & Baselines

To rigorously evaluate FlyMind, five navigation controllers are benchmarked under identical physical conditions:

1. **FlyMindBio (Full Pipeline)**:
   Integrated HR-EMD $\to$ LPLC2 $\to$ GF reflex + WTA gap finding + flow divergence nearness + IMU derotation + Safety Supervisor.
2. **LPLC2-GF Only**:
   Looming-triggered reflex only (no gap-seeking steering; forward cruise with emergency braking climb).
3. **WTA Flow Gap Only**:
   Lateral flow minimization steering only (no looming-triggered saccades).
4. **Canny Edge Density Baseline**:
   Conventional computer vision baseline steering away from high edge-gradient concentrations.
5. **Monocular DepthNet Baseline**:
   Deep learning baseline predicting monocular inverse depth to steer toward maximal clearance.

---

## 4. Quantitative Evaluation Metrics

All trials are executed across 50 pseudo-randomized seeds in three challenging simulation arenas:
1. **Collision Frequency ($F_{\text{coll}}$)**: Percentage of trials resulting in contact ($d_{\text{min}} < r_{\text{drone}}$).
2. **Minimum Obstacle Clearance ($d_{\text{clear}}$)**: Closest spatial distance to any obstacle over the flight path ($m$).
3. **Safe-Navigation Completion Rate ($R_{\text{comp}}$)**: Percentage of trials reaching the target boundary without collision.
4. **Processing Latency ($T_{\text{proc}}$)**: Mean and 99th-percentile computational runtime per perception-action cycle (ms).
5. **Computational Load**: CPU utilization and memory footprint.
6. **Lighting Robustness**: Performance degradation under nominal, low-light (reduced SNR), and high-contrast glare conditions.

---

## 5. Software Modularity & Safety Invariants

* **Flight Control Decoupling**: FlyMind computes high-level velocity commands $(v_x, v_y, v_z, \dot{\psi})$. Low-level motor mixing and attitude stabilization are preserved within dedicated flight controller firmware.
* **Independent Safety Interlock**: The `SafetySupervisor` enforces hard limits on speed ($3.0\text{ m/s}$), yaw rate ($120^\circ/\text{s}$), altitude ($[0.3\text{ m}, 3.0\text{ m}]$), and executes a 200 ms watchdog failsafe hover if perception drops below $10\text{ Hz}$.
