"""
LPLC2 (Lobula Plate / Lobula Columnar 2) Neural Modeling Subpackage.
Reproduces Zhao et al. (iScience 2023) 4-lobed radial receptive fields and multiplicative integration.
"""
from .cross_kernel import LPLC2ReceptiveField, LPLC2Subunit
from .population_coding import LPLC2Population

__all__ = ["LPLC2ReceptiveField", "LPLC2Subunit", "LPLC2Population"]
