"""
Task-agnostic registry of attention architectures.

This package owns the single mapping from architecture name (as used in
notebook/experiment configs) to its implementation class. Adding a new
architecture means adding a module in this package and one entry here; task
models and notebooks then see it automatically. Task code must not define
its own architecture-name mappings.

Registered names:
- ``softmax``      scaled dot-product attention with softmax-normalized weights
- ``linear``       the same scoring without the final softmax normalization
- ``strassen``     retained high-order variant (three bilinear score matrices)
- ``third_order``  retained high-order variant (triples of tokens)
- ``triangular``   retained pair-level variant; operates on ``(B, N, N, C)``
                    and is therefore **not sequence-composable** through
                    ``EncoderLayer``

All architectures build on the shared ``BaseAttention`` configuration
interface (head count plus independently configurable query/key and value head
dimensions). The high-order variants are retained for future work.
"""

from compgen.models.attentions.softmax import SoftmaxAttention
from compgen.models.attentions.linear import LinearAttention
from compgen.models.attentions.strassen import StrassenAttention
from compgen.models.attentions.triangular import TriangularAttention
from compgen.models.attentions.third_order import ThirdOrderAttention

ATTENTION_CLASSES = {
    "softmax": SoftmaxAttention,
    "linear": LinearAttention,
    "strassen": StrassenAttention,
    "triangular": TriangularAttention,
    "third_order": ThirdOrderAttention,
}
