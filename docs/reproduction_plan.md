# FlyMind: Scientific Reproduction Plan & Development Milestones

This document specifies the scientific reproduction plan for extracting, reimplementing, and validating core bio-inspired algorithms from open-source literature within the **FlyMind** research platform.

---

## 1. Reproduction Mapping & Scientific Baselines

### Module A: EMD $\to$ LPLC2 $\to$ Giant Fiber Looming Detector
* **Scientific Reference**: Junyu Zhao et al., *iScience* 2023 ([PMC10073940](https://pmc.ncbi.nlm.nih.gov/articles/PMC10073940/)).
* **Original Implementation**: MATLAB script [`upstream/EMD-LPLC2-GF-model/`](file:///d:/Projects/Fruit%20fly/upstream/EMD-LPLC2-GF-model).
* **FlyMind Reproduction Strategy**:
  - Reimplement 2D spatial delay-and-correlate Hassenstein-Reichardt EMDs using vectorized PyTorch tensor operations.
  - Implement 4-lobed cross-kernel convolution representing LPLC2 receptive fields ($80^\circ \times 80^\circ$ equivalent angular scope).
  - Apply multiplicative nonlinear integration $\prod_{k=1}^4 [L_k - L_0]_+$ across the 4 cardinal directions to guarantee rejection of laminar background translation.
  - Formulate the population activity curve $N_{\text{act}}(t)$ and drive a Leaky Integrate-and-Fire (LIF) Giant Fiber neuron with threshold $\theta_{\text{trigger}} \approx 34^\circ$.
* **Validation Test**: Verify that the module fires for approaching (expanding) objects and remains silent for laminar drifting grating stimuli.

---

### Module B: Neuromorphic Winner-Take-All Gap Selector
* **Scientific Reference**: Jesse J. Hagenaars, Guido C. H. E. de Croon et al., *Nature Communications* 2024.
* **FlyMind Reproduction Strategy**:
  - Partition the forward $120^\circ$ field of view into $M = 16$ angular sectors.
  - Compute optical flow magnitude within each sector $H_m$.
  - Implement lateral inhibitory Winner-Take-All dynamics to identify the sector with minimal optical flow (the most distant open gap).
  - Compute a yaw rate setpoint balancing the flow minimum direction with bilateral lateral flow centering.
* **Validation Test**: Navigate through a slit corridor with narrow gates without colliding with either post.

---

### Module C: IMU-Derotated Angular Acceleration Filter
* **Scientific Reference**: Jiannan Zhao et al., *Biomimetics* / *Drones* 2024.
* **FlyMind Reproduction Strategy**:
  - Ingest 3-axis gyro rates $[p, q, r]$ from the simulated IMU.
  - Analytically subtract rotational optic flow $\mathbf{u}_{\text{rot}} = \mathbf{B}(\mathbf{x}) \boldsymbol{\omega}$ from total flow to isolate pure translational flow $\mathbf{u}_{\text{trans}}$.
  - Compute visual angular acceleration $\ddot{\theta}(t)$ of expanding obstacle edges to distinguish genuine collision hazards from vehicle maneuvers.
* **Validation Test**: Execute aggressive quadrotor yaw/bank turns while in open air; verify zero false looming alarms.

---

### Module D: Flow Divergence Relative Nearness Map
* **Scientific Reference**: *Neurocomputing* (2024).
* **FlyMind Reproduction Strategy**:
  - Compute 2D spatial flow divergence $\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$ across the frame using central finite differences.
  - Normalize by forward speed $v_z$ to compute relative nearness $\hat{\mu}(\mathbf{x})$.
  - Generate a 2D proximity heatmap and estimate instantaneous Time-to-Contact ($\text{TTC} = \frac{1}{\nabla \cdot \mathbf{u}}$).
* **Validation Test**: Move the drone toward a planar wall; verify that flow divergence increases linearly as distance decreases.

---

### Module E: Independent Safety Supervisor
* **Architectural Reference**: SpikeCalls, *FlyDrones* (2026).
* **FlyMind Reproduction Strategy**:
  - Implement a hardware-isolated supervisor that wraps high-level navigation commands.
  - Enforce hard bounding limits: maximum linear speed ($3.0\text{ m/s}$), maximum yaw rate ($120^\circ/\text{s}$), ceiling ($3.0\text{ m}$), and safety floor ($0.3\text{ m}$).
  - Implement watchdog timer: if the neural model drops below $10\text{ Hz}$ or outputs invalid numbers, the supervisor commands an immediate safe hover state.
* **Validation Test**: Send corrupted/infinite velocity commands from the navigation module; verify that the supervisor intercepts and forces an immediate stable hover.

---

## 2. Development Milestones

```text
Milestone 1: Simulation Foundation & World Arenas
├── 6-DoF quadrotor dynamics (RK4 Newton-Euler)
├── Synthetic forward camera with depth buffer & noisy IMU
└── 3 reproducible clutter environments (forest, gates, looming hazard)

Milestone 2: Perception & Nearness Mapping
├── Optical flow engines (Lucas-Kanade & Farnebäck)
├── 2D spatial flow divergence filter
└── Dense relative nearness map generator

Milestone 3: Bio-Inspired Neural Circuitry
├── Vectorized 2D Hassenstein-Reichardt EMD array
├── 4-lobed multiplicative LPLC2 convolution
├── Spiking Giant Fiber (GF / DNp01) trigger module
└── Neuromorphic Winner-Take-All (WTA) gap selector

Milestone 4: Safe Navigation & Hardware-Level Supervisor
├── Reactive navigation decision system (gap steering + velocity modulation)
├── Reflexive escape saccade state machine
└── Isolated Safety Supervisor with geofencing and emergency hover/landing

Milestone 5: Baselines & Comparative Benchmark Suite
├── Classical OpenCV edge-density baseline
├── Monocular depth / collision ML baseline
└── Automated 50-seed benchmark runner (collision rate, clearance, latency)

Milestone 6: Visual Research Dashboard & Documentation
├── Live browser research console (video stream, vector flow, nearness heatmap, HUD)
├── 100% passing unit & integration test suite
└── Final research report and documentation
```

---

## 3. Ready for Execution

All preliminary analyses, code investigations, and scientific models are complete. Upon your approval, we will proceed immediately with **Milestone 1** to establish the `flymind/` codebase and simulation environment.
