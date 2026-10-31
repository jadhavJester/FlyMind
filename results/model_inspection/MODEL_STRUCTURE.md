# BioFlight: FruitFly MuJoCo Model Architecture & Structure

This document summarizes the reverse-engineered kinematics, dynamics, actuators, and sensor configurations of the FruitFly model in `flybody`.

## 1. Flight Model Overview

- **Total Bodies**: 135
- **Total Joints**: 38
- **Position DoFs (nq)**: 50
- **Velocity DoFs (nv)**: 48
- **Actuators**: 11 (flight configuration with leg DoFs retracted/disabled)
- **Action Space**: Box(12,)
- **Observation Dimension**: 104 across 9 observables
- **Physics Timestep**: `5e-05 s` (20000 Hz)
- **Control Timestep**: `0.0002 s` (5000 Hz)

### Flight Actuators
| Actuator Name | Control Range |
|---|---|
| `walker/head_abduct` | [-0.2, 0.2] |
| `walker/head_twist` | [-3.0, 3.0] |
| `walker/head` | [-0.5, 0.3] |
| `walker/wing_yaw_left` | [-1.0, 1.0] |
| `walker/wing_roll_left` | [-1.0, 1.0] |
| `walker/wing_pitch_left` | [-1.0, 1.0] |
| `walker/wing_yaw_right` | [-1.0, 1.0] |
| `walker/wing_roll_right` | [-1.0, 1.0] |
| `walker/wing_pitch_right` | [-1.0, 1.0] |
| `walker/abdomen_abduct` | [-0.7, 0.7] |
| `walker/abdomen` | [-1.05, 0.7] |

## 2. Walking Model Comparison

- **Total Bodies**: 135
- **Total Joints**: 104
- **Actuators**: 59 (all 6 articulated legs active; wings retracted)
- **Action Space**: Box(59,)
- **Observation Dimension**: 741
- **Physics Timestep**: `0.0002 s` (5000 Hz)
- **Control Timestep**: `0.002 s` (500 Hz)

## 3. FlyBody Kinematic Hierarchy (Body Tree)

```text
└── world
    ├── walker/
    │   └── walker/thorax
    │       ├── walker/head
    │       │   ├── walker/rostrum
    │       │   │   └── walker/haustellum
    │       │   │       ├── walker/labrum_left
    │       │   │       └── walker/labrum_right
    │       │   ├── walker/antenna_left
    │       │   └── walker/antenna_right
    │       ├── walker/wing_left
    │       ├── walker/wing_right
    │       ├── walker/abdomen
    │       │   └── walker/abdomen_2
    │       │       └── walker/abdomen_3
    │       │           └── walker/abdomen_4
    │       │               └── walker/abdomen_5
    │       │                   └── walker/abdomen_6
    │       │                       └── walker/abdomen_7
    │       ├── walker/haltere_left
    │       ├── walker/haltere_right
    │       ├── walker/coxa_T1_left
    │       │   └── walker/femur_T1_left
    │       │       └── walker/tibia_T1_left
    │       │           └── walker/tarsus_T1_left
    │       │               └── walker/tarsus2_T1_left
    │       │                   └── walker/tarsus3_T1_left
    │       │                       └── walker/tarsus4_T1_left
    │       │                           └── walker/claw_T1_left
    │       ├── walker/coxa_T1_right
    │       │   └── walker/femur_T1_right
    │       │       └── walker/tibia_T1_right
    │       │           └── walker/tarsus_T1_right
    │       │               └── walker/tarsus2_T1_right
    │       │                   └── walker/tarsus3_T1_right
    │       │                       └── walker/tarsus4_T1_right
    │       │                           └── walker/claw_T1_right
    │       ├── walker/coxa_T2_left
    │       │   └── walker/femur_T2_left
    │       │       └── walker/tibia_T2_left
    │       │           └── walker/tarsus_T2_left
    │       │               └── walker/tarsus2_T2_left
    │       │                   └── walker/tarsus3_T2_left
    │       │                       └── walker/tarsus4_T2_left
    │       │                           └── walker/claw_T2_left
    │       ├── walker/coxa_T2_right
    │       │   └── walker/femur_T2_right
    │       │       └── walker/tibia_T2_right
    │       │           └── walker/tarsus_T2_right
    │       │               └── walker/tarsus2_T2_right
    │       │                   └── walker/tarsus3_T2_right
    │       │                       └── walker/tarsus4_T2_right
    │       │                           └── walker/claw_T2_right
    │       ├── walker/coxa_T3_left
    │       │   └── walker/femur_T3_left
    │       │       └── walker/tibia_T3_left
    │       │           └── walker/tarsus_T3_left
    │       │               └── walker/tarsus2_T3_left
    │       │                   └── walker/tarsus3_T3_left
    │       │                       └── walker/tarsus4_T3_left
    │       │                           └── walker/claw_T3_left
    │       └── walker/coxa_T3_right
    │           └── walker/femur_T3_right
    │               └── walker/tibia_T3_right
    │                   └── walker/tarsus_T3_right
    │                       └── walker/tarsus2_T3_right
    │                           └── walker/tarsus3_T3_right
    │                               └── walker/tarsus4_T3_right
    │                                   └── walker/claw_T3_right
    └── ghost/
        └── ghost/thorax
            ├── ghost/head
            │   ├── ghost/rostrum
            │   │   └── ghost/haustellum
            │   │       ├── ghost/labrum_left
            │   │       └── ghost/labrum_right
            │   ├── ghost/antenna_left
            │   └── ghost/antenna_right
            ├── ghost/abdomen
            │   └── ghost/abdomen_2
            │       └── ghost/abdomen_3
            │           └── ghost/abdomen_4
            │               └── ghost/abdomen_5
            │                   └── ghost/abdomen_6
            │                       └── ghost/abdomen_7
            ├── ghost/haltere_left
            ├── ghost/haltere_right
            ├── ghost/coxa_T1_left
            │   └── ghost/femur_T1_left
            │       └── ghost/tibia_T1_left
            │           └── ghost/tarsus_T1_left
            │               └── ghost/tarsus2_T1_left
            │                   └── ghost/tarsus3_T1_left
            │                       └── ghost/tarsus4_T1_left
            │                           └── ghost/claw_T1_left
            ├── ghost/coxa_T1_right
            │   └── ghost/femur_T1_right
            │       └── ghost/tibia_T1_right
            │           └── ghost/tarsus_T1_right
            │               └── ghost/tarsus2_T1_right
            │                   └── ghost/tarsus3_T1_right
            │                       └── ghost/tarsus4_T1_right
            │                           └── ghost/claw_T1_right
            ├── ghost/coxa_T2_left
            │   └── ghost/femur_T2_left
            │       └── ghost/tibia_T2_left
            │           └── ghost/tarsus_T2_left
            │               └── ghost/tarsus2_T2_left
            │                   └── ghost/tarsus3_T2_left
            │                       └── ghost/tarsus4_T2_left
            │                           └── ghost/claw_T2_left
            ├── ghost/coxa_T2_right
            │   └── ghost/femur_T2_right
            │       └── ghost/tibia_T2_right
            │           └── ghost/tarsus_T2_right
            │               └── ghost/tarsus2_T2_right
            │                   └── ghost/tarsus3_T2_right
            │                       └── ghost/tarsus4_T2_right
            │                           └── ghost/claw_T2_right
            ├── ghost/coxa_T3_left
            │   └── ghost/femur_T3_left
            │       └── ghost/tibia_T3_left
            │           └── ghost/tarsus_T3_left
            │               └── ghost/tarsus2_T3_left
            │                   └── ghost/tarsus3_T3_left
            │                       └── ghost/tarsus4_T3_left
            │                           └── ghost/claw_T3_left
            └── ghost/coxa_T3_right
                └── ghost/femur_T3_right
                    └── ghost/tibia_T3_right
                        └── ghost/tarsus_T3_right
                            └── ghost/tarsus2_T3_right
                                └── ghost/tarsus3_T3_right
                                    └── ghost/tarsus4_T3_right
                                        └── ghost/claw_T3_right
```

## 4. Observables Breakdown (Flight Model)

| Observable Name | Shape |
|---|---|
| `walker/accelerometer` | [3] |
| `walker/actuator_activation` | [0] |
| `walker/gyro` | [3] |
| `walker/joints_pos` | [25] |
| `walker/joints_vel` | [25] |
| `walker/velocimeter` | [3] |
| `walker/world_zaxis` | [3] |
| `walker/ref_displacement` | [6, 3] |
| `walker/ref_root_quat` | [6, 4] |
