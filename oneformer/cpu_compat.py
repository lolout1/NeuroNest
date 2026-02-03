"""
CPU compatibility utilities for OneFormer on PyTorch CPU builds.
Provides fallbacks for torch.cuda.amp components when CUDA is not available.
"""
import torch
from contextlib import contextmanager
from functools import wraps

if torch.cuda.is_available():
    # Use real CUDA autocast when GPU is available
    from torch.cuda.amp import autocast
    try:
        from torch.cuda.amp import custom_bwd, custom_fwd
    except ImportError:
        # Fallback if custom_bwd/custom_fwd don't exist
        def custom_fwd(**kwargs):
            def decorator(func):
                return func
            return decorator

        def custom_bwd(**kwargs):
            def decorator(func):
                return func
            return decorator
else:
    # CPU-only fallbacks

    @contextmanager
    def autocast(enabled=True, dtype=None, cache_enabled=None):
        """
        No-op autocast context manager for CPU inference.
        Maintains API compatibility with torch.cuda.amp.autocast.
        """
        yield

    def custom_fwd(**kwargs):
        """
        No-op decorator for custom forward pass on CPU.
        Returns the function unchanged.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def custom_bwd(**kwargs):
        """
        No-op decorator for custom backward pass on CPU.
        Returns the function unchanged.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

# Monkey patch torch.cuda.amp and torch.amp if needed
if not torch.cuda.is_available():
    import sys
    from types import ModuleType

    # Patch torch.cuda.amp
    if not hasattr(torch.cuda, 'amp'):
        torch.cuda.amp = ModuleType('amp')
        sys.modules['torch.cuda.amp'] = torch.cuda.amp

    torch.cuda.amp.autocast = autocast
    torch.cuda.amp.custom_fwd = custom_fwd
    torch.cuda.amp.custom_bwd = custom_bwd

    # Patch torch.amp (in case imports come from there)
    if not hasattr(torch, 'amp'):
        torch.amp = ModuleType('amp')
        sys.modules['torch.amp'] = torch.amp

    # Inject fallbacks into torch.amp as well
    if not hasattr(torch.amp, 'autocast'):
        torch.amp.autocast = autocast
    if not hasattr(torch.amp, 'custom_fwd'):
        torch.amp.custom_fwd = custom_fwd
    if not hasattr(torch.amp, 'custom_bwd'):
        torch.amp.custom_bwd = custom_bwd

__all__ = ['autocast', 'custom_fwd', 'custom_bwd']
