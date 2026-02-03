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
        def custom_fwd(_func=None, **kwargs):
            def decorator(func):
                return func
            if _func is not None:
                return _func
            return decorator

        def custom_bwd(_func=None, **kwargs):
            def decorator(func):
                return func
            if _func is not None:
                return _func
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

    def custom_fwd(_func=None, **kwargs):
        """
        No-op decorator for custom forward pass on CPU.
        Supports both @custom_fwd and @custom_fwd() calling styles.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        # Handle @custom_fwd (without parentheses)
        if _func is not None:
            return _func
        # Handle @custom_fwd() or @custom_fwd(cast_inputs=...)
        return decorator

    def custom_bwd(_func=None, **kwargs):
        """
        No-op decorator for custom backward pass on CPU.
        Supports both @custom_bwd and @custom_bwd() calling styles.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        # Handle @custom_bwd (without parentheses)
        if _func is not None:
            return _func
        # Handle @custom_bwd() or @custom_bwd(cast_inputs=...)
        return decorator

# Monkey patch torch modules for CPU compatibility
import sys
from types import ModuleType

# Always patch torch.nn.attention if it doesn't exist (PyTorch 1.12.1 doesn't have it)
if not hasattr(torch.nn, 'attention'):
    # Create torch.nn.attention package
    torch.nn.attention = ModuleType('torch.nn.attention')
    sys.modules['torch.nn.attention'] = torch.nn.attention

    # Create torch.nn.attention.flex_attention submodule
    flex_attention = ModuleType('torch.nn.attention.flex_attention')
    sys.modules['torch.nn.attention.flex_attention'] = flex_attention
    torch.nn.attention.flex_attention = flex_attention

# Patch torch.cuda.amp and torch.amp if on CPU
if not torch.cuda.is_available():
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
