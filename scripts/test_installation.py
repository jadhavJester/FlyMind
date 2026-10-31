"""
Test installation script for the BioFlight project.
Verifies Python version, package installations (FlyBody, MuJoCo, dm_control, NumPy),
optional ML frameworks (PyTorch, TensorFlow), system compute (CPU/GPU),
and simulation physics stepping.
"""

import os
import platform
import sys

def test_component(name, check_fn):
    try:
        details, passed = check_fn()
        status = "PASS" if passed else "FAIL"
        return name, status, details
    except Exception as e:
        return name, "FAIL", str(e)

def check_python():
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    passed = (v.major == 3 and v.minor == 10)
    return f"Python {ver_str} ({platform.python_implementation()})", passed

def check_system():
    info = f"{platform.system()} {platform.release()} ({platform.machine()}) | CPU: {platform.processor() or 'Unknown'}"
    return info, True

def check_numpy():
    import numpy as np
    ver = np.__version__
    passed = ver.startswith("1.26")
    return f"v{ver}", passed

def check_mujoco():
    import mujoco
    ver = mujoco.__version__
    # Run a simple physics check
    xml = "<mujoco><worldbody><geom type='plane' size='1 1 0.1'/><body pos='0 0 1'><joint type='free'/><geom type='sphere' size='0.1'/></body></worldbody></mujoco>"
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d)
    return f"v{ver} (step verified)", True

def check_dm_control():
    import dm_control
    from dm_control import suite
    env = suite.load(domain_name="cartpole", task_name="swingup")
    time_step = env.reset()
    passed = time_step.step_type is not None
    ver = getattr(dm_control, "__version__", "installed")
    return f"v{ver} (cartpole test passed)", passed

def check_flybody():
    import flybody
    from flybody.fly_envs import walk_imitation
    # Verify instantiation of a baseline fly environment without external dataset
    env = walk_imitation()
    ts = env.reset()
    import numpy as np
    action = np.zeros(env.action_spec().shape, dtype=np.float32)
    env.step(action)
    ver = getattr(flybody, "__version__", "0.1.0")
    return f"v{ver} (walk_imitation instantiated & stepped)", True

def check_gpu():
    # Detect PyTorch / TF / DirectML or report CPU execution
    gpu_found = False
    details = "No dedicated GPU detected (CPU mode enabled)"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_found = True
            details = f"CUDA GPU: {torch.cuda.get_device_name(0)}"
    except ImportError:
        pass
    return details, True

def check_pytorch():
    try:
        import torch
        return f"v{torch.__version__}", True
    except ImportError:
        return "Not installed (Optional)", True

def check_tensorflow():
    try:
        import tensorflow as tf
        return f"v{tf.__version__}", True
    except ImportError:
        return "Not installed (Optional)", True

def main():
    print("=" * 70)
    print(" BioFlight — System & Dependency Verification")
    print("=" * 70)

    checks = [
        ("Operating System & CPU", check_system),
        ("Python Version (3.10)", check_python),
        ("NumPy (1.26.x required)", check_numpy),
        ("MuJoCo Physics", check_mujoco),
        ("dm_control", check_dm_control),
        ("FlyBody & FruitFly Env", check_flybody),
        ("Compute / GPU", check_gpu),
        ("PyTorch", check_pytorch),
        ("TensorFlow", check_tensorflow),
    ]

    results = []
    all_passed = True

    for name, fn in checks:
        comp_name, status, details = test_component(name, fn)
        results.append((comp_name, status, details))
        if status == "FAIL":
            all_passed = False

    # Print Table
    col1 = 28
    col2 = 8
    print(f"{'Component':<{col1}} | {'Status':<{col2}} | Details")
    print("-" * 70)
    for name, status, details in results:
        print(f"{name:<{col1}} | {status:<{col2}} | {details}")
    print("=" * 70)

    if all_passed:
        print("OVERALL STATUS: PASS — All core dependencies are operational!")
        sys.exit(0)
    else:
        print("OVERALL STATUS: FAIL — Some dependencies failed inspection.")
        sys.exit(1)

if __name__ == "__main__":
    main()
