"""
CPU compatibility utilities for OneFormer on PyTorch CPU builds.
Provides fallback for torch.cuda.amp.autocast when CUDA is not available.
"""
import torch
from contextlib import contextmanager

if torch.cuda.is_available():
    # Use real CUDA autocast when GPU is available
    from torch.cuda.amp import autocast
else:
    # CPU-only fallback: no-op context manager
    @contextmanager
    def autocast(enabled=True, dtype=None, cache_enabled=None):
        """
        No-op autocast context manager for CPU inference.
        Maintains API compatibility with torch.cuda.amp.autocast.
        """
        yield

__all__ = ['autocast']
