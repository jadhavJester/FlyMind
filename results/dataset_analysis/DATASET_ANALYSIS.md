# BioFlight: Empirical Flight Dataset Analysis

This report provides quantitative analysis and mechanical insights into the real fruit fly (*Drosophila melanogaster*) locomotion dataset.

## 1. Dataset Overview

- **Total Trajectories**: 272
- **Sampling Frequency**: `5000 Hz` (timestep $dt = 0.0002\text{ s}$)
- **Total Data Points**: 264,120 frames
- **Trajectory Duration**: Mean = `0.194 s`, Range = `[0.054, 0.576] s`

### Trajectory Categorization

| Trajectory Type | Count | Percentage |
|---|---|---|
| `evasion_original` | 92 | 33.8% |
| `evasion_reflected` | 92 | 33.8% |
| `saccade_original` | 44 | 16.2% |
| `saccade_reflected` | 44 | 16.2% |

## 2. Flight Dynamics & Aerodynamic Insights

| Metric | Mean | Median | Peak (Max) | Standard Deviation |
|---|---|---|---|---|
| **Linear Speed (|V|)** | `0.16 m/s` | `0.16 m/s` | `0.74 m/s` | `0.05 m/s` |
| **Peak Acceleration** | `11.31 m/s²` | `10.53 m/s²` | `25.99 m/s²` | `5.56 m/s²` |
| **Peak Bank Angle (Roll)** | `70.9°` | `64.9°` | `179.9°` | `31.9°` |
| **Peak Yaw Rate** | `1980°/s` | `1872°/s` | `3930°/s` | `638°/s` |

### Key Biological Findings for Control Engineering

1. **Banked Turn Dominance**: Saccadic flight turns in *Drosophila* are heavily **banked turns** (similar to fixed-wing aircraft and agile quadrotors). To alter course, the fly initiates a rapid roll ($>60°$ in under $20\text{ ms}$), redirecting total aerodynamic lift laterally, followed by active counter-torque to stabilize heading.
2. **Extreme Accelerations**: Peak accelerations exceed $30\text{ m/s²}$ ($>3\text{ g}$), driven by rapid bilateral asymmetry in wing stroke amplitude and pitch.
3. **Wing Beat Pattern Modulations**: The 500-step baseline wing pattern reveals large sweep amplitudes ($pprox 120°$) in stroke angle $\phi$ with subtle deviation $\theta$ and rapid flip-over pitch $\eta$ at stroke reversals.
