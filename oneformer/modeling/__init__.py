from .backbone.swin import D2SwinTransformer

# DiNAT requires natten library with PyTorch 2.x features
# Make it optional since only Swin is used in production
try:
    from .backbone.dinat import D2DiNAT
    DINAT_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"DiNAT backbone not available (natten library issue): {e}")
    D2DiNAT = None
    DINAT_AVAILABLE = False

from .pixel_decoder.fpn import BasePixelDecoder
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .meta_arch.oneformer_head import OneFormerHead
