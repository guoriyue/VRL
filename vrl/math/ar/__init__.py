"""Autoregressive math shared by replay evaluators and training code."""

from vrl.math.ar.logprob import gather_categorical_log_probs

__all__ = ["gather_categorical_log_probs"]
