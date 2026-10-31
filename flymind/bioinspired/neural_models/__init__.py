"""
Neural Models Subpackage
Spiking and rate-based neurons implementing Drosophila descending and command circuits.
"""
from .giant_fiber_lif import GiantFiberNeuron, GFSpikeState

__all__ = ["GiantFiberNeuron", "GFSpikeState"]
