# FlyDrones: Fruit Fly Connectome Drone Pilot Integration

[![FlyDrones](https://img.shields.io/badge/FlyDrones-Integrated-brightgreen.svg)](https://github.com/SpikeCalls/FlyDrones)
[![Tests](https://img.shields.io/badge/tests-30%2F30%20passing-brightgreen.svg)](upstream/FlyDrones/tests/)
[![Connectome](https://img.shields.io/badge/connectome-MaleCNS%20v1.0%20%7C%20MiniFly-blue.svg)](https://male-cns.janelia.org/)

This document outlines the architecture, installation, verification, and interactive usage of **FlyDrones** ([github.com/SpikeCalls/FlyDrones](https://github.com/SpikeCalls/FlyDrones)), now fully cloned, built, and verified inside the BioFlight project workspace.

---

## 1. Overview & Conceptual Breakthrough

In 2026, the complete central nervous system of a male fruit fly (*Drosophila melanogaster*) was published: **MaleCNS v1.0** (~166,000 neurons, ~25.6M directed synaptic connections).

**FlyDrones** bridges computational neurobiology and robotics:
```
Camera Feed / Hand Gesture
       │
       ▼
Fly Retinal Flow & Looming (R1-R6, T4/T5, LPLC2, LC4)
       │
       ▼
Leaky Integrate-and-Fire (LIF) Connectome (166k MaleCNS or 850 MiniFly)
       │
       ▼
Descending Motor Neurons (DNg02, DNp01, DNp03)
       │
       ▼
Safety Governor (Altitude limits, slew rates, watchdog)
       │
       ▼
Drone Flight Setpoints (Throttle, Yaw Rate, Forward, Escape)
```

The drone's pilot is not an arbitrary machine-learning model; it is the **biological connectome itself**, where sensory illusions (optic flow drift and looming expansion) trigger natural behavioral reflexes.

---

## 2. Neural Circuit Mapping

| Stage | Biological Neurons | Natural Function in *Drosophila* | FlyDrones Drone Mapping |
|---|---|---|---|
| **Photoreceptors** | `R1-R6` | Ommatidial brightness sensing | Downsampled image brightness per eye |
| **Optic Flow** | `T4a-d` / `T5a-d` | Motion direction (front-to-back, back-to-front, up, down) | Lucas-Kanade optical flow vector field |
| **Lobula Plate** | `HS`, `VS`, `LPi` | Horizontal & vertical wide-field integration | Rotational yaw & vertical sink/climb cues |
| **Looming Detectors**| `LPLC2`, `LC4` | Approaching/expanding threat detection | Optical expansion rate triggers escape |
| **Gyroscopic Feedback**| `Haltere` afferents | Body angular velocity sensing | Drone IMU angular velocity feedback |
| **Wing-Stroke Control**| `DNg02` (L/R) | Controls left & right wing-beat amplitudes | $Throttle \propto L + R$, $Yaw \propto R - L$ |
| **Flight Saccades** | `DNp03` (L/R) | Rapid evasive steering turns | Evasive turning away from looming threats |
| **Giant Fiber Escape** | `DNp01` (L/R) | Explosive jump-takeoff reflex | High-thrust escape surge / altitude jump |

---

## 3. Verified Build & Test Suite

FlyDrones is installed in editable mode under `upstream/FlyDrones/` in the active environment.

All **30/30 unit & integration tests** are passing:
```bash
pytest upstream/FlyDrones/tests/
# Output: 30 passed in upstream/FlyDrones/tests/
```

Test coverage includes:
- Connectome graph parsing and group resolution (`test_brain.py`)
- Sensorimotor core subgraph extraction (`test_brain.py`)
- Leaky Integrate-and-Fire (LIF) spiking physics (`test_lif.py`)
- Optic flow, looming detection, and gesture illusions (`test_senses.py`)
- Motor decoding, readouts, and safety governors (`test_control.py`)
- MaleCNS Feather dataset loader (`test_malecns_loader.py`)
- Closed-loop simulation and CLI commands (`test_sim_and_cli.py`)

---

## 4. How to Run FlyDrones

### 1. Interactive 3D Web Simulation (Three.js WebGL)
A custom launcher serves the 3D bedroom environment, animated fly-drone model, and live neural spiking visualizer:
```bash
python scripts/run_flydrones_web.py
```
* **Browser URL**: `http://localhost:8088`
* **Controls**:
  - `S`: **Swat It** — swings a fly swatter at the drone; looming detector fires and triggers giant-fiber escape.
  - `1` – `6`: Trigger optical-flow illusions (Open palm = climb, Fist = hover hold, Left/Right = optomotor turn, Rush = escape, Drop = descend).
  - Click on the **Brain** panel to stimulate individual neural circuits directly.
  - `C`: Cycle camera view (Orbit, Chase, Drone Onboard Camera).
  - `N`: Toggle Day / Night lighting.

### 2. Connectome Inspection & Circuit Stimulation
Inspect all input and output neuron groups and run a 500 ms electrical poke test:
```bash
python -m flydrones inspect
```

### 3. Closed-Loop Simulated Flight
Run simulated flight with synthetic hand-gesture illusions:
```bash
# Terminal log output for 5 seconds
python -m flydrones demo --seconds 5

# Full simulation with video/GIF output
python -m flydrones demo --record videos/flydrones_demo.gif
```

### 4. Swarm Simulation (1 Connectome, 3 Drone Pilots)
Deploy three independent drone pilots sharing the identical underlying connectome wiring but processing individual sensory inputs and noise:
```bash
python -m flydrones swarm --seconds 10
```

---

## 5. Synergies with BioFlight UAV Framework

BioFlight contains 6-DoF quadrotor dynamics in [`simulation/uav/quadrotor_dynamics.py`](file:///d:/Projects/Fruit%20fly/simulation/uav/quadrotor_dynamics.py) and high-level controllers in [`src/controllers/uav_controllers.py`](file:///d:/Projects/Fruit%20fly/src/controllers/uav_controllers.py). 

With FlyDrones now available, descending neuron firing rates (`DNg02`, `DNp01`, `DNp03`) can directly govern the motor thrust allocations of BioFlight quadrotors, providing a direct link between biological connectomics and high-fidelity aerodynamic simulation.
