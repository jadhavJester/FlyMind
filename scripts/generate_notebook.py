"""Helper script to construct notebooks/01_dataset_exploration.ipynb cleanly."""

import json
import os

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# BioFlight: 01 — Real Flight Trajectory Dataset Exploration\n",
                "\n",
                "This notebook performs exploratory data analysis on the empirical high-speed fruit fly (*Drosophila melanogaster*) locomotion dataset.\n",
                "- **Dataset**: `data/raw/flight-dataset_saccade-evasion_augmented.hdf5`\n",
                "- **Source**: Science 2014 & J Exp Biol 2015 (Florian T. Muijres et al., HHMI Janelia)\n",
                "- **Sampling**: 5,000 Hz ($dt = 0.0002\\text{ s}$)\n",
                "- **Total Trajectories**: 272 saccades and evasion maneuvers"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from mpl_toolkits.mplot3d import Axes3D\n",
                "\n",
                "# Add project root to sys.path\n",
                "project_root = os.path.abspath(os.path.join(os.getcwd(), \"..\"))\n",
                "if project_root not in sys.path:\n",
                "    sys.path.insert(0, project_root)\n",
                "\n",
                "from src.data.flight_dataset import FlightDataset\n",
                "print(\"BioFlight data module successfully imported!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Dataset & Summary Statistics"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dataset_path = os.path.join(project_root, \"data\", \"raw\", \"flight-dataset_saccade-evasion_augmented.hdf5\")\n",
                "wing_pattern_path = os.path.join(project_root, \"data\", \"raw\", \"wing_pattern_fmech.npy\")\n",
                "\n",
                "dataset = FlightDataset(dataset_path, wing_pattern_path)\n",
                "print(f\"Total trajectories loaded: {len(dataset)}\")\n",
                "print(f\"Control frequency: {dataset.freq_hz:.0f} Hz (dt={dataset.dt} s)\")\n",
                "print(f\"Trajectory categories: {list(dataset.type_indices.keys())}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. 3D Trajectory Visualization"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig = plt.figure(figsize=(10, 7))\n",
                "ax = fig.add_subplot(111, projection='3d')\n",
                "\n",
                "colors = {'evasion_original': '#E63946', 'evasion_reflected': '#F4A261',\n",
                "          'saccade_original': '#2A9D8F', 'saccade_reflected': '#457B9D'}\n",
                "\n",
                "for i in range(min(20, len(dataset))):\n",
                "    traj = dataset[i]\n",
                "    pos_mm = traj.position_cm * 10.0  # cm to mm\n",
                "    c = colors.get(traj.traj_type, '#333333')\n",
                "    ax.plot(pos_mm[:, 0], pos_mm[:, 1], pos_mm[:, 2], color=c, alpha=0.7, lw=1.5)\n",
                "    ax.scatter(pos_mm[0, 0], pos_mm[0, 1], pos_mm[0, 2], color='green', s=10)\n",
                "    ax.scatter(pos_mm[-1, 0], pos_mm[-1, 1], pos_mm[-1, 2], color='red', s=10)\n",
                "\n",
                "ax.set_title(\"3D Flight Saccades & Evasions (mm)\", fontsize=12, fontweight='bold')\n",
                "ax.set_xlabel(\"X (mm)\")\n",
                "ax.set_ylabel(\"Y (mm)\")\n",
                "ax.set_zlabel(\"Z (mm)\")\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Position (X, Y, Z), Velocity, and Acceleration vs. Time"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "traj = dataset[0]\n",
                "t_ms = traj.timestamps * 1000.0\n",
                "pos_mm = traj.position_cm * 10.0\n",
                "\n",
                "fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)\n",
                "axes[0].plot(t_ms, pos_mm[:, 0], label='X (mm)', color='#2B6CB0')\n",
                "axes[0].plot(t_ms, pos_mm[:, 1], label='Y (mm)', color='#C53030')\n",
                "axes[0].plot(t_ms, pos_mm[:, 2], label='Z (mm)', color='#2F855A')\n",
                "axes[0].set_ylabel(\"Position (mm)\")\n",
                "axes[0].legend(loc='upper left', ncol=3)\n",
                "axes[0].grid(True)\n",
                "\n",
                "axes[1].plot(t_ms, traj.linear_velocity[:, 0], label='Vx (m/s)', color='#2B6CB0')\n",
                "axes[1].plot(t_ms, traj.linear_velocity[:, 1], label='Vy (m/s)', color='#C53030')\n",
                "axes[1].plot(t_ms, traj.linear_velocity[:, 2], label='Vz (m/s)', color='#2F855A')\n",
                "axes[1].plot(t_ms, traj.speed, label='|V|', color='#2D3748', linestyle='--')\n",
                "axes[1].set_ylabel(\"Velocity (m/s)\")\n",
                "axes[1].legend(loc='upper left', ncol=4)\n",
                "axes[1].grid(True)\n",
                "\n",
                "axes[2].plot(t_ms, traj.linear_acceleration[:, 0], label='Ax (m/s²)', color='#2B6CB0')\n",
                "axes[2].plot(t_ms, traj.linear_acceleration[:, 1], label='Ay (m/s²)', color='#C53030')\n",
                "axes[2].plot(t_ms, traj.linear_acceleration[:, 2], label='Az (m/s²)', color='#2F855A')\n",
                "axes[2].plot(t_ms, traj.accel_magnitude, label='|A|', color='#805AD5', linestyle='--')\n",
                "axes[2].set_ylabel(\"Accel (m/s²)\")\n",
                "axes[2].set_xlabel(\"Time (ms)\")\n",
                "axes[2].legend(loc='upper left', ncol=4)\n",
                "axes[2].grid(True)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Attitude: Roll, Pitch, Yaw and Banked Turn Dynamics"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)\n",
                "axes[0].plot(t_ms, traj.euler_deg[:, 0], label='Roll (bank angle)', color='#D69E2E', lw=2)\n",
                "axes[0].plot(t_ms, traj.euler_deg[:, 1], label='Pitch', color='#319795', lw=1.8)\n",
                "axes[0].plot(t_ms, traj.euler_deg[:, 2], label='Yaw (heading)', color='#805AD5', lw=1.8)\n",
                "axes[0].set_ylabel(\"Attitude (deg)\")\n",
                "axes[0].set_title(f\"Trajectory #{traj.traj_id} — Rotational Kinematics\", fontweight='bold')\n",
                "axes[0].legend(loc='upper left', ncol=3)\n",
                "axes[0].grid(True)\n",
                "\n",
                "axes[1].plot(t_ms, np.rad2deg(traj.angular_velocity[:, 0]), label='Roll rate (deg/s)', color='#D69E2E')\n",
                "axes[1].plot(t_ms, np.rad2deg(traj.angular_velocity[:, 1]), label='Pitch rate (deg/s)', color='#319795')\n",
                "axes[1].plot(t_ms, np.rad2deg(traj.angular_velocity[:, 2]), label='Yaw rate (deg/s)', color='#805AD5')\n",
                "axes[1].set_ylabel(\"Angular Rate (deg/s)\")\n",
                "axes[1].set_xlabel(\"Time (ms)\")\n",
                "axes[1].legend(loc='upper left', ncol=3)\n",
                "axes[1].grid(True)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Wing Beat Kinematics (Pattern Generator)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if dataset.wing_pattern is not None:\n",
                "    wp = dataset.wing_pattern\n",
                "    phase = np.linspace(0, 1.0, len(wp))\n",
                "    plt.figure(figsize=(8, 4))\n",
                "    plt.plot(phase, np.rad2deg(wp[:, 0]), label='Stroke Angle $\\phi$ (Yaw)', color='#E53E3E', lw=2)\n",
                "    plt.plot(phase, np.rad2deg(wp[:, 1]), label='Deviation Angle $\\theta$ (Roll)', color='#3182CE', lw=2)\n",
                "    plt.plot(phase, np.rad2deg(wp[:, 2]), label='Wing Pitch $\\eta$', color='#38A169', lw=2)\n",
                "    plt.title(\"Baseline Wing Beat Pattern (1 Flapping Period)\", fontweight='bold')\n",
                "    plt.xlabel(\"Normalized Flapping Phase\")\n",
                "    plt.ylabel(\"Angle (degrees)\")\n",
                "    plt.legend(loc='upper right')\n",
                "    plt.grid(True)\n",
                "    plt.tight_layout()\n",
                "    plt.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.20"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = os.path.join("notebooks", "01_dataset_exploration.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Generated {out_path}")
