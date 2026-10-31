"""
Model inspection script for BioFlight (Phase 3).
Reverse-engineers the FlyBody MuJoCo model:
- Number of bodies, body names, parent-child tree
- Joints (names, types, ranges, limits)
- Actuators (names, types, control ranges)
- Sensors (names, types, dims)
- Simulation timesteps and frequencies
Saves:
- results/model_inspection/model_summary.json
- results/model_inspection/MODEL_STRUCTURE.md
"""

import os
import json
import numpy as np
from flybody.fly_envs import flight_imitation, walk_imitation

JOINT_TYPE_NAMES = {
    0: "free",
    1: "ball",
    2: "slide",
    3: "hinge"
}

SENSOR_TYPE_NAMES = {
    0: "touch",
    1: "accelerometer",
    2: "velocimeter",
    3: "gyro",
    4: "force",
    5: "torque",
    6: "magnetometer",
    7: "rangefinder",
    8: "jointpos",
    9: "jointvel",
    10: "tendonpos",
    11: "tendonvel",
    12: "actuatorpos",
    13: "actuatorvel",
    14: "actuatorfrc",
    15: "ballquat",
    16: "ballangvel",
    17: "jointlimitpos",
    18: "jointlimitvel",
    19: "jointlimitfrc",
    20: "tendonlimitpos",
    21: "tendonlimitvel",
    22: "tendonlimitfrc",
    23: "framepos",
    24: "framequat",
    25: "framexaxis",
    26: "frameyaxis",
    27: "framezaxis",
    28: "framelinvel",
    29: "frameangvel",
    30: "framelinacc",
    31: "frameangacc",
    32: "subtreepos",
    33: "subtreelinvel",
    34: "subtreeangmom",
    35: "user"
}

def inspect_env_physics(env_name, env):
    physics = env.physics
    model = physics.model.ptr
    data = physics.data.ptr

    # 1. Timestep & frequency
    timestep = float(model.opt.timestep)
    ctrl_timestep = float(env.control_timestep())
    sim_freq = 1.0 / timestep if timestep > 0 else 0.0
    ctrl_freq = 1.0 / ctrl_timestep if ctrl_timestep > 0 else 0.0

    # 2. Bodies & hierarchy
    n_bodies = model.nbody
    body_names = [physics.model.id2name(i, 'body') or f"body_{i}" for i in range(n_bodies)]
    body_parents = [int(model.body_parentid[i]) for i in range(n_bodies)]

    # Build hierarchy tree dict
    children_map = {i: [] for i in range(n_bodies)}
    for i in range(1, n_bodies):
        parent_id = body_parents[i]
        children_map[parent_id].append(i)

    def build_tree(body_id):
        return {
            "name": body_names[body_id],
            "id": body_id,
            "children": [build_tree(child_id) for child_id in children_map[body_id]]
        }
    hierarchy_tree = build_tree(0)

    # 3. Joints
    n_joints = model.njnt
    joints_info = []
    for j in range(n_joints):
        name = physics.model.id2name(j, 'joint') or f"joint_{j}"
        j_type = int(model.jnt_type[j])
        j_type_name = JOINT_TYPE_NAMES.get(j_type, f"unknown_{j_type}")
        limited = bool(model.jnt_limited[j])
        range_min = float(model.jnt_range[j][0]) if limited else None
        range_max = float(model.jnt_range[j][1]) if limited else None
        body_id = int(model.jnt_bodyid[j])
        joints_info.append({
            "name": name,
            "type": j_type_name,
            "body": body_names[body_id],
            "limited": limited,
            "range": [range_min, range_max] if limited else None
        })

    # 4. Actuators
    n_actuators = model.nu
    actuators_info = []
    for a in range(n_actuators):
        name = physics.model.id2name(a, 'actuator') or f"actuator_{a}"
        ctrl_min = float(model.actuator_ctrlrange[a][0])
        ctrl_max = float(model.actuator_ctrlrange[a][1])
        actuators_info.append({
            "name": name,
            "ctrl_range": [ctrl_min, ctrl_max]
        })

    # 5. Sensors
    n_sensors = model.nsensor
    sensors_info = []
    for s in range(n_sensors):
        name = physics.model.id2name(s, 'sensor') or f"sensor_{s}"
        s_type = int(model.sensor_type[s])
        s_type_name = SENSOR_TYPE_NAMES.get(s_type, f"type_{s_type}")
        s_dim = int(model.sensor_dim[s])
        sensors_info.append({
            "name": name,
            "type": s_type_name,
            "dim": s_dim
        })

    # 6. Observables & Action Spec
    obs_spec = env.observation_spec()
    action_spec = env.action_spec()
    observables = {k: list(v.shape) for k, v in obs_spec.items()}

    summary = {
        "env_name": env_name,
        "n_bodies": n_bodies,
        "n_joints": n_joints,
        "n_dof": int(model.nv),
        "n_qpos": int(model.nq),
        "n_actuators": n_actuators,
        "n_sensors": n_sensors,
        "physics_timestep_s": timestep,
        "physics_freq_hz": sim_freq,
        "control_timestep_s": ctrl_timestep,
        "control_freq_hz": ctrl_freq,
        "action_dimension": int(action_spec.shape[0]),
        "total_obs_dimension": sum(int(np.prod(s)) for s in observables.values()),
        "observables": observables,
        "body_names": body_names,
        "hierarchy_tree": hierarchy_tree,
        "joints": joints_info,
        "actuators": actuators_info,
        "sensors": sensors_info
    }
    return summary

def format_ascii_tree(node, prefix="", is_last=True):
    lines = []
    connector = "└── " if is_last else "├── "
    name = node["name"]
    lines.append(prefix + connector + name)
    
    new_prefix = prefix + ("    " if is_last else "│   ")
    children = node.get("children", [])
    for idx, child in enumerate(children):
        last_child = (idx == len(children) - 1)
        lines.extend(format_ascii_tree(child, new_prefix, last_child))
    return lines

def main():
    out_dir = os.path.join("results", "model_inspection")
    os.makedirs(out_dir, exist_ok=True)

    # Inspect Flight Model
    print("Inspecting Flight Imitation Model...")
    flight_env = flight_imitation()
    flight_summary = inspect_env_physics("flight_imitation", flight_env)

    # Inspect Walking Model
    print("Inspecting Walking Imitation Model...")
    walk_env = walk_imitation()
    walk_summary = inspect_env_physics("walk_imitation", walk_env)

    combined_summary = {
        "flight_model": flight_summary,
        "walk_model": walk_summary
    }

    # Save JSON summary
    json_path = os.path.join(out_dir, "model_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2)
    print(f"Saved JSON summary to {json_path}")

    # Generate MODEL_STRUCTURE.md
    md_path = os.path.join(out_dir, "MODEL_STRUCTURE.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# BioFlight: FruitFly MuJoCo Model Architecture & Structure\n\n")
        f.write("This document summarizes the reverse-engineered kinematics, dynamics, actuators, and sensor configurations of the FruitFly model in `flybody`.\n\n")
        
        f.write("## 1. Flight Model Overview\n\n")
        f.write(f"- **Total Bodies**: {flight_summary['n_bodies']}\n")
        f.write(f"- **Total Joints**: {flight_summary['n_joints']}\n")
        f.write(f"- **Position DoFs (nq)**: {flight_summary['n_qpos']}\n")
        f.write(f"- **Velocity DoFs (nv)**: {flight_summary['n_dof']}\n")
        f.write(f"- **Actuators**: {flight_summary['n_actuators']} (flight configuration with leg DoFs retracted/disabled)\n")
        f.write(f"- **Action Space**: Box({flight_summary['action_dimension']},)\n")
        f.write(f"- **Observation Dimension**: {flight_summary['total_obs_dimension']} across {len(flight_summary['observables'])} observables\n")
        f.write(f"- **Physics Timestep**: `{flight_summary['physics_timestep_s']} s` ({flight_summary['physics_freq_hz']:.0f} Hz)\n")
        f.write(f"- **Control Timestep**: `{flight_summary['control_timestep_s']} s` ({flight_summary['control_freq_hz']:.0f} Hz)\n\n")

        f.write("### Flight Actuators\n")
        f.write("| Actuator Name | Control Range |\n|---|---|\n")
        for act in flight_summary["actuators"]:
            f.write(f"| `{act['name']}` | [{act['ctrl_range'][0]}, {act['ctrl_range'][1]}] |\n")

        f.write("\n## 2. Walking Model Comparison\n\n")
        f.write(f"- **Total Bodies**: {walk_summary['n_bodies']}\n")
        f.write(f"- **Total Joints**: {walk_summary['n_joints']}\n")
        f.write(f"- **Actuators**: {walk_summary['n_actuators']} (all 6 articulated legs active; wings retracted)\n")
        f.write(f"- **Action Space**: Box({walk_summary['action_dimension']},)\n")
        f.write(f"- **Observation Dimension**: {walk_summary['total_obs_dimension']}\n")
        f.write(f"- **Physics Timestep**: `{walk_summary['physics_timestep_s']} s` ({walk_summary['physics_freq_hz']:.0f} Hz)\n")
        f.write(f"- **Control Timestep**: `{walk_summary['control_timestep_s']} s` ({walk_summary['control_freq_hz']:.0f} Hz)\n\n")

        f.write("## 3. FlyBody Kinematic Hierarchy (Body Tree)\n\n")
        f.write("```text\n")
        tree_lines = format_ascii_tree(flight_summary["hierarchy_tree"])
        f.write("\n".join(tree_lines))
        f.write("\n```\n\n")

        f.write("## 4. Observables Breakdown (Flight Model)\n\n")
        f.write("| Observable Name | Shape |\n|---|---|\n")
        for k, v in flight_summary["observables"].items():
            f.write(f"| `{k}` | {v} |\n")

    print(f"Saved Markdown documentation to {md_path}")

if __name__ == "__main__":
    main()
