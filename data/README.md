# BioFlight — Dataset Documentation

## 1. Official Dataset Sources

The empirical locomotion data supporting `flybody` originates from high-speed motion tracking experiments conducted at HHMI Janelia Research Campus:

| Dataset Name | Source / DOI | Key / Figshare URL | Compressed Size | Uncompressed Format |
|---|---|---|---|---|
| **Flight Imitation Dataset** | [10.25378/janelia.25309105](https://doi.org/10.25378/janelia.25309105) | [File 51196859](https://janelia.figshare.com/ndownloader/files/51196859) | ~1.4 GB | HDF5 (`.hdf5`) |
| **Walking Imitation Dataset** | [10.25378/janelia.25309105](https://doi.org/10.25378/janelia.25309105) | [File 51196868](https://janelia.figshare.com/ndownloader/files/51196868) | ~1.2 GB | HDF5 (`.hdf5`) |
| **Pretrained Checkpoints** | [10.25378/janelia.25309105](https://doi.org/10.25378/janelia.25309105) | [File 44815195](https://janelia.figshare.com/ndownloader/files/44815195) | ~800 MB | TF / Pickle Checkpoints |
| **Local Augmented Trajectories** | Included in FlySuite | `upstream/FlySuite/data/` | ~109 MB | HDF5 & NumPy (`.npy`) |

### Included Starter Datasets
For local CPU development and initial offline inspection, lightweight verified HDF5 trajectory files are immediately available in `upstream/FlySuite/data/`:
1. `flight-dataset_saccade-evasion_augmented.hdf5` (14.3 MB): Contains full multi-trajectory 3D positions, orientations (quaternions), velocities, and timestamps of fruit-fly rapid saccadic evasion maneuvers.
2. `walking-dataset-small_female-only_snippets-100_min-len-0.5s_trk-files-0-9.hdf5` (94.8 MB): High-speed walking pose snippets.
3. `wing_pattern_fmech.npy` (6.1 KB): Baseline wing beat kinematic pattern for the Wing Beat Pattern Generator.

---

## 2. Dataset Structure & Format

The trajectory datasets are stored in standard **HDF5** (`.hdf5`) hierarchy:

```text
file.hdf5
├── 'timestep_seconds': float scalar (e.g. 0.0002 s for flight, 0.002 s for walking)
└── 'trajectories' (Group)
    ├── '0'
    │   ├── 'position': shape (T, 3) [x, y, z in meters]
    │   ├── 'quaternion': shape (T, 4) [w, x, y, z orientation]
    │   ├── 'velocity': shape (T, 3) [vx, vy, vz in m/s]
    │   ├── 'angular_velocity': shape (T, 3) [wx, wy, wz in rad/s]
    │   └── 'joint_angles': shape (T, N_joints)
    ├── '1'
    └── ...
```

---

## 3. Automated Download Procedure

To download full research-scale datasets from Janelia Figshare into `data/raw/`, run:

```bash
# In Python:
from flybody.download_data import figshare_download

# Download flight imitation dataset:
figshare_download('flight-imitation-dataset', dest_path='data/raw')

# Download walking imitation dataset:
figshare_download('walking-imitation-dataset', dest_path='data/raw')
```

Or via shell:
```bash
python scripts/download_dataset.py --dataset flight
```

---

## 4. Storage & Git Policy
- Raw datasets (`data/raw/*.hdf5`) and large processed artifacts (`data/processed/*.hdf5`) are excluded from Git via `.gitignore`.
- Use `data/README.md` to document any custom processed splits and preprocessing metadata.
