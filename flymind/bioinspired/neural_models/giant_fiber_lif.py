"""
Giant Fiber (GF) Leaky Integrate-and-Fire (LIF) Escape Neuron Model.
Reproduces the invariant angular threshold trigger (theta ~ 34 deg) observed in Drosophila looming escape circuits.

Neurodynamics:
    tau_m * dV/dt = -(V - V_rest) + R_m * I_syn(t)
    When V >= V_thresh:
        Fire action potential (spike = True)
        V <- V_reset
        Enter absolute refractory period t_ref
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class GFSpikeState:
    """State of the Giant Fiber neuron at a discrete timestep."""
    membrane_potential_mv: float
    is_spiking: bool
    is_refractory: bool
    synaptic_current_pA: float
    time_since_last_spike_s: float


class GiantFiberNeuron:
    """
    Simulates the Drosophila Giant Fiber escape neuron as a LIF system.
    Receives excitatory synaptic input from LPLC2 population activity.
    """
    def __init__(
        self,
        v_rest_mv: float = -65.0,
        v_thresh_mv: float = -45.0,
        v_reset_mv: float = -70.0,
        tau_m_s: float = 0.015,       # 15 ms membrane time constant
        r_m_mohm: float = 100.0,      # 100 MOhm membrane resistance
        refractory_period_s: float = 0.020,  # 20 ms refractory period
        weight_lplc2: float = 2.5,    # Synaptic weight per active LPLC2 unit (pA)
    ):
        self.v_rest = v_rest_mv
        self.v_thresh = v_thresh_mv
        self.v_reset = v_reset_mv
        self.tau_m = tau_m_s
        self.r_m = r_m_mohm
        self.refractory_period = refractory_period_s
        self.weight_lplc2 = weight_lplc2

        self.v = self.v_rest
        self._time_since_spike: float = 999.0

    def reset(self) -> None:
        """Reset membrane potential to rest."""
        self.v = self.v_rest
        self._time_since_spike = 999.0

    def step(self, n_act: int, lplc2_mean_act: float, dt: float = 0.02) -> GFSpikeState:
        """
        Integrate one simulation timestep.
        Args:
            n_act: Number of active LPLC2 units (N_act).
            lplc2_mean_act: Mean activation of LPLC2 population.
            dt: Timestep in seconds.
        Returns:
            GFSpikeState reporting membrane potential and spike status.
        """
        self._time_since_spike += dt
        is_refractory = self._time_since_spike < self.refractory_period

        # Compute synaptic current from LPLC2 inputs (pA)
        # Driven by both number of active units and their collective firing intensity
        i_syn = (n_act * self.weight_lplc2) + (lplc2_mean_act * 5.0)

        spike = False
        if not is_refractory:
            # LIF integration: dV = ( -(V - V_rest) + R_m * I_syn ) * (dt / tau_m)
            # Scaling factor so current in pA with R_m in MOhm produces realistic mV shifts
            dv = (-(self.v - self.v_rest) + (self.r_m * i_syn * 0.01)) * (dt / self.tau_m)
            self.v += dv

            if self.v >= self.v_thresh:
                spike = True
                self.v = self.v_reset
                self._time_since_spike = 0.0
        else:
            # Clamped at reset potential during refractory period
            self.v = self.v_reset

        return GFSpikeState(
            membrane_potential_mv=float(self.v),
            is_spiking=spike,
            is_refractory=is_refractory,
            synaptic_current_pA=float(i_syn),
            time_since_last_spike_s=float(self._time_since_spike),
        )
