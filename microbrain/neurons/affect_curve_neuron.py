from __future__ import annotations

"""
Compatibility tombstone for Hormone Curve Field v1.1.

v1 briefly used a parallel affect_curve_neuron to publish curve telemetry.  v1.1
folds curve math directly into reward_novelty_pulse_neuron so that the existing
reward/novelty organ remains the only authoritative publisher of affect/reward.

This module intentionally exposes no build_neurons() function.  The auto-loader
will import it and skip it, preventing duplicate affect/reward authority if an
older v1 file existed in the tree.
"""

PATCH_NOTE = "Hormone Curve Field v1.1: curve math is integrated into reward_novelty_pulse_neuron."
