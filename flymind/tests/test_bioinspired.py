"""
Test Bio-Inspired Neural Modules: LPLC2 Cross-Kernel, Population Array, Giant Fiber LIF, and WTA Gap Selector.
"""
import numpy as np
import pytest

from flymind.bioinspired.lplc2.cross_kernel import LPLC2ReceptiveField
from flymind.bioinspired.lplc2.population_coding import LPLC2Population
from flymind.bioinspired.neural_models.giant_fiber_lif import GiantFiberNeuron
from flymind.bioinspired.motion_detectors.wta_gap import WinnerTakeAllGapSelector


def test_lplc2_receptive_field_quenches_uniform_translation():
    rf = LPLC2ReceptiveField(center_x=30, center_y=30, radius=20, inhibition_alpha=0.25)
    # Uniform rightward translation: right direction is active everywhere, but other 3 directions are 0
    emd_signals = {
        "right": np.ones((60, 60), dtype=np.float32) * 5.0,
        "left": np.zeros((60, 60), dtype=np.float32),
        "up": np.zeros((60, 60), dtype=np.float32),
        "down": np.zeros((60, 60), dtype=np.float32),
    }
    sub = rf.compute(emd_signals)
    # Multiplicative integration must yield 0 for uniform laminar translation!
    assert sub.net_activation == 0.0


def test_lplc2_receptive_field_fires_on_expansion():
    rf = LPLC2ReceptiveField(center_x=30, center_y=30, radius=20, inhibition_alpha=0.10)
    # Looming divergence: all 4 directions active in their outward quadrants
    emd_signals = {
        "right": np.ones((60, 60), dtype=np.float32) * 5.0,
        "left": np.ones((60, 60), dtype=np.float32) * 5.0,
        "up": np.ones((60, 60), dtype=np.float32) * 5.0,
        "down": np.ones((60, 60), dtype=np.float32) * 5.0,
    }
    sub = rf.compute(emd_signals)
    assert sub.net_activation > 0.0


def test_giant_fiber_lif_neuron():
    gf = GiantFiberNeuron(v_rest_mv=-65.0, v_thresh_mv=-45.0)
    # Sub-threshold step
    s1 = gf.step(n_act=1, lplc2_mean_act=0.1, dt=0.02)
    assert not s1.is_spiking
    assert s1.membrane_potential_mv < -45.0

    # Strong looming barrage should depolarize above threshold and fire
    spiked = False
    for _ in range(10):
        s = gf.step(n_act=25, lplc2_mean_act=5.0, dt=0.02)
        if s.is_spiking:
            spiked = True
            break
    assert spiked, "Giant Fiber should spike under strong population looming activation"


def test_wta_gap_selector():
    wta = WinnerTakeAllGapSelector(num_sectors=8, fov_deg=90.0)
    # Simulate corridor where sector 3 is an open window (zero obstacle density)
    # and all other sectors have dense obstacle clutter
    clutter = np.ones(8, dtype=np.float32) * 4.0
    clutter[3] = 0.0  # open gap

    decision = wta.select_gap(clutter)
    assert decision.best_sector_idx == 3
    assert decision.sector_confidence > 0.0
