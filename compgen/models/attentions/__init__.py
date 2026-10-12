"""
Attention architectures provided by the model layer.

This package holds the concrete attention implementations. There is no
name->architecture registry and no name->constructed-module helper: every
caller that needs an attention module imports the concrete class from here (or
from its submodule) and instantiates the object directly with the
configuration it needs. Adding an architecture means adding a module in this
package and constructing that class where it is used.

Available architectures:
- ``SoftmaxAttention``    scaled dot-product attention with softmax-normalized
                          weights
- ``LinearAttention``     the same scoring without the final softmax
                          normalization
- ``StrassenAttention``   retained high-order variant (three bilinear score
                          matrices)
- ``ThirdOrderAttention`` retained high-order variant (triples of tokens)
- ``TriangularAttention`` retained pair-level variant; operates on
                          ``(B, N, N, C)`` and is therefore **not
                          sequence-composable** through ``EncoderLayer``

All architectures build on the shared ``BaseAttention`` configuration
interface (head count plus independently configurable query/key and value head
dimensions). The high-order variants are retained for future work.
"""

from compgen.models.attentions.softmax import SoftmaxAttention
from compgen.models.attentions.linear import LinearAttention
from compgen.models.attentions.strassen import StrassenAttention
from compgen.models.attentions.triangular import TriangularAttention
from compgen.models.attentions.third_order import ThirdOrderAttention
