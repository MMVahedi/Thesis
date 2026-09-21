"""
Task-agnostic registry of attention architectures.

This package owns the single mapping from architecture name (as used in
notebook/experiment configs) to its implementation class. Adding a new
architecture means adding a module in this package and one entry here; task
models and notebooks then see it automatically. Task code must not define
its own architecture-name mappings.
"""

from compgen.models.attentions.standard import StandardAttention
from compgen.models.attentions.strassen import StrassenAttention
from compgen.models.attentions.triangular import TriangularAttention
from compgen.models.attentions.third_order import ThirdOrderAttention

ATTENTION_CLASSES = {
    "standard": StandardAttention,
    "strassen": StrassenAttention,
    "triangular": TriangularAttention,
    "third_order": ThirdOrderAttention,
}