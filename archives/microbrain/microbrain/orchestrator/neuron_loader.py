from __future__ import annotations

import importlib
import pkgutil
import logging
from typing import Iterable

from .orchestrator import Orchestrator
from .neuron_base import BaseNeuron

logger = logging.getLogger(__name__)


def auto_register_neurons(
    orchestrator: Orchestrator,
    base_package: str = "microbrain.neurons",
) -> None:
    """
    Scan a neurons package, import each module, and let it register
    its neurons with the orchestrator.

    Convention for each neuron module:

        # microbrain/neurons/some_neuron.py

        from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig

        class MyNeuron(BaseNeuron):
            ...

        def build_neurons(orchestrator: Orchestrator) -> Iterable[BaseNeuron]:
            cfg = NeuronConfig(
                name="my_neuron",
                subscribed_topics=["percept/text"],
                output_topics=["act/speech"],
            )
            yield MyNeuron(cfg)

    Any module in `base_package` that defines a callable
    `build_neurons(orchestrator)` can return one or more `BaseNeuron`
    instances, which will be registered automatically.
    """
    try:
        pkg = importlib.import_module(base_package)
    except ImportError:
        # If the neurons package doesn't exist yet, that's fine;
        # we just don't register anything.
        logger.warning("Neurons package %r not found; no neurons registered.", base_package)
        return

    # Iterate over all modules directly under the neurons package
    for finder, mod_name, is_pkg in pkgutil.iter_modules(
        pkg.__path__, pkg.__name__ + "."
    ):
        try:
            module = importlib.import_module(mod_name)
        except Exception as e:
            # Bad neuron module shouldn't kill the system; log and continue.
            logger.exception("Error importing neuron module %r: %s", mod_name, e)
            continue

        build_fn = getattr(module, "build_neurons", None)
        if not callable(build_fn):
            continue

        try:
            neurons = list(build_fn(orchestrator))
        except Exception as e:
            # Make failures VERY visible during debugging
            logger.exception(
                "Error while building neurons from module %r: %s", mod_name, e
            )
            continue

        for neuron in neurons:
            if not isinstance(neuron, BaseNeuron):
                logger.warning(
                    "Module %r returned non-BaseNeuron instance %r; skipping.",
                    mod_name,
                    neuron,
                )
                continue

            # Let the orchestrator handle uniqueness & bus registration
            orchestrator.register_neuron(neuron)
