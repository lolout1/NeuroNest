import torch
from contextlib import contextmanager


class AttentionCaptureHook:
    """Captures attention weights from EomtAttention.forward() output[1]."""

    def __init__(self):
        self.weights = None
        self._h = None

    def _fn(self, module, args, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            self.weights = output[1].detach().cpu()

    def register(self, module):
        self._h = module.register_forward_hook(self._fn)
        return self

    def remove(self):
        if self._h:
            self._h.remove()
            self._h = None


class HiddenStateCaptureHook:
    """Captures hidden states (first element of layer output)."""

    def __init__(self):
        self.state = None
        self._h = None

    def _fn(self, module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        self.state = out.detach().cpu()

    def register(self, module):
        self._h = module.register_forward_hook(self._fn)
        return self

    def remove(self):
        if self._h:
            self._h.remove()
            self._h = None


class ActivationGradientHook:
    """Captures forward activations AND backward gradients."""

    def __init__(self):
        self.activation = None
        self.gradient = None
        self._fh = None
        self._bh = None

    def register(self, module):
        self._fh = module.register_forward_hook(
            lambda m, i, o: setattr(
                self, "activation", o[0] if isinstance(o, tuple) else o
            )
        )
        self._bh = module.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradient", go[0])
        )
        return self

    def remove(self):
        if self._fh:
            self._fh.remove()
        if self._bh:
            self._bh.remove()
        self._fh = self._bh = None


@contextmanager
def eager_attention(model):
    """Temporarily switch to eager attention so weights are returned (SDPA returns None)."""
    orig = getattr(model.config, "_attn_implementation", "sdpa")
    model.config._attn_implementation = "eager"
    try:
        yield
    finally:
        model.config._attn_implementation = orig
