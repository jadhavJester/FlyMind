# Embodied Alphabet Spelling Mission: 'UDIT', 'FLY', 'AI'

This document details the extension of the FlyBody embodied task architecture to sequential spatial target navigation across a 26-letter alphabet arena ($A \dots Z$).

---

## 1. Overview & Objective

In this task:
- 26 spatial points labeled with the English alphabet (**'A' through 'Z'**) are scattered across the 3D flight arena ($12\text{ cm} \times 12\text{ cm} \times 5.8\text{ cm}$).
- The fruit fly (*Drosophila melanogaster*) is commanded to physically navigate through the arena and spell target words by contacting letters in order.
- The primary target word is **"UDIT"** (spelling $\text{U} \to \text{D} \to \text{I} \to \text{T}$), followed by **"FLY"** and **"AI"**.

---

## 2. Alphabet Arena Geometry (26 Letters Scattered)

Each letter has a unique 3D position in the arena, with a contact radius of $r = 0.15\text{ cm}$ ($1.5\text{ mm}$):

| Letter | Spatial Coordinates $[X, Y, Z]\text{ cm}$ | Role |
| :---: | :---: | :--- |
| **U** | $[+0.60, +0.25, 0.55]$ | 1st Letter in "UDIT" |
| **D** | $[+0.60, -0.25, 0.55]$ | 2nd Letter in "UDIT" |
| **I** | $[+0.55, +0.12, 0.60]$ | 3rd Letter in "UDIT" / 2nd in "AI" |
| **T** | $[+0.55, -0.12, 0.60]$ | 4th Letter in "UDIT" |
| **F** | $[+0.65, +0.18, 0.52]$ | 1st Letter in "FLY" |
| **L** | $[+0.65, -0.18, 0.52]$ | 2nd Letter in "FLY" |
| **Y** | $[+0.50, +0.28, 0.65]$ | 3rd Letter in "FLY" |
| **A** | $[+0.50, -0.28, 0.65]$ | 1st Letter in "AI" |
| **B..Z** | Scattered in $X \in [0.45, 0.75], Y \in [-0.35, +0.35], Z \in [0.50, 0.70]$ | Background Alphabet Points |

---

## 3. Experimental Results

The spelling script [`scripts/run_spelling_task.py`](file:///d:/Projects/Fruit%20fly/scripts/run_spelling_task.py) executed all spelling missions in live MuJoCo physics:

### Word Mission 1: "UDIT" (Completed Successfully)
* Target sequence: **U** $\to$ **D** $\to$ **I** $\to$ **T**

| Letter | Target Position $[X, Y, Z]\text{ cm}$ | Steps | Flight Time | Min Distance to Letter | Contact Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **U** | $[+0.60, +0.25, 0.55]$ | 179 | $35.8\text{ ms}$ | **$0.147\text{ cm}$** | **ACQUIRED** ($d \le 0.150\text{ cm}$) |
| **D** | $[+0.60, -0.25, 0.55]$ | 194 | $38.8\text{ ms}$ | **$0.145\text{ cm}$** | **ACQUIRED** ($d \le 0.150\text{ cm}$) |
| **I** | $[+0.55, +0.12, 0.60]$ | 162 | $32.4\text{ ms}$ | **$0.150\text{ cm}$** | **ACQUIRED** ($d \le 0.150\text{ cm}$) |
| **T** | $[+0.55, -0.12, 0.60]$ | 173 | $34.6\text{ ms}$ | **$0.150\text{ cm}$** | **ACQUIRED** ($d \le 0.150\text{ cm}$) |

**Result**: Word **"UDIT"** was 100% spelled and collected in 708 cumulative control steps ($141.6\text{ ms}$ total flight time).

### Word Mission 2: "FLY"
* **F**: Acquired at step 183 (min dist: $0.150\text{ cm}$).
* **L**: Near-miss ($0.204\text{ cm}$).
* **Y**: Acquired at step 158 (min dist: $0.148\text{ cm}$).

### Word Mission 3: "AI"
* **A**: Acquired at step 167 (min dist: $0.149\text{ cm}$).
* **I**: Acquired at step 163 (min dist: $0.148\text{ cm}$).

---

## 4. Visual Artifacts

1. **Alphabet Arena & Trajectory Map**:
   - File: [`results/spelling/alphabet_arena_spelling.png`](file:///d:/Projects/Fruit%20fly/results/spelling/alphabet_arena_spelling.png)
   - Shows all 26 letter positions with "UDIT" highlighted in gold and connected by directional flight vectors.
2. **Spelling Telemetry Video**:
   - File: [`videos/fly_spelling_words.mp4`](file:///d:/Projects/Fruit%20fly/videos/fly_spelling_words.mp4) (851 frames @ 30 fps)
   - File: [`videos/fly_spelling_words.webp`](file:///d:/Projects/Fruit%20fly/videos/fly_spelling_words.webp) (animated WebP)
   - Dual-camera view displaying the 3D tracking perspective alongside the top-down arena view showing all 26 scattered alphabet letters, real-time target seeking, and spelled letter acquisition banners.
3. **Structured Logs**:
   - File: [`results/spelling/spelling_logs.json`](file:///d:/Projects/Fruit%20fly/results/spelling/spelling_logs.json)
