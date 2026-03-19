"""
Named configuration presets for grubeta.

Presets map finance concepts to model parameters so users don't need
to understand GRU units, dropout, or epochs.

Usage:
    >>> from grubeta.presets import get_preset, list_presets
    >>> config = get_preset("responsive")
    >>> list_presets()
"""

from grubeta.core import DynamicBetaConfig

PRESETS = {
    "default": DynamicBetaConfig(
        lookback=60, initial_train_size=500, wf_step_size=126,
        gru_units=64, lambda_beta=0.05, lambda_alpha=0.5,
        lambda_alpha_smooth=0.1, verbose=1
    ),
    "responsive": DynamicBetaConfig(
        lookback=30, initial_train_size=252, wf_step_size=21,
        gru_units=32, lambda_beta=0.02, lambda_alpha=0.3,
        lambda_alpha_smooth=0.05, verbose=1
    ),
    "smooth": DynamicBetaConfig(
        lookback=120, initial_train_size=756, wf_step_size=252,
        gru_units=64, lambda_beta=0.15, lambda_alpha=0.5,
        lambda_alpha_smooth=0.15, verbose=1
    ),
    "research": DynamicBetaConfig(
        lookback=90, initial_train_size=500, wf_step_size=63,
        gru_units=128, epochs_init=60, epochs_retrain=6,
        lambda_beta=0.05, lambda_alpha=0.5,
        lambda_alpha_smooth=0.1, verbose=1
    ),
}

PRESET_DESCRIPTIONS = {
    "default": "Balanced settings for most use cases (3-month lookback, quarterly retraining)",
    "responsive": "Captures rapid beta changes (1-month lookback, monthly retraining) — good for event studies, earnings reactions",
    "smooth": "Stable estimates for long-term portfolio construction (6-month lookback, annual retraining)",
    "research": "Enhanced model capacity for academic analysis (128 GRU units, longer training, finer walk-forward steps)",
}


def get_preset(name: str) -> DynamicBetaConfig:
    """Get a named configuration preset. Returns a copy so it can be modified."""
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{name}'. Available presets: {available}\n"
            f"Use grubeta.list_presets() to see descriptions."
        )
    return PRESETS[name].model_copy()


def list_presets() -> dict:
    """List available presets with descriptions."""
    return PRESET_DESCRIPTIONS.copy()
