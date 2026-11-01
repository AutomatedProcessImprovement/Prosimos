"""
Each engine, pick and catalogue draws from generators of its own, seeded from the seed and its name, so that one
doesn't change the others' draws. Prosimos draws from the global generators, so an engine's states are put there
while it is called.
"""
import hashlib
import random

import numpy as np


def engine_random_states(seed, process_name):
    """Initial Python and NumPy generator states for one engine. A stable hash of the seed and the
    process name, not Python's hash(), which differs between runs."""
    if seed is None:
        python_seed, numpy_seed = random.getrandbits(64), random.getrandbits(32)
    else:
        digest = hashlib.sha256(f"{seed}/{process_name}".encode("utf-8")).digest()
        python_seed, numpy_seed = int.from_bytes(digest[:8], "big"), int.from_bytes(digest[8:12], "big")
    return random.Random(python_seed).getstate(), np.random.RandomState(numpy_seed).get_state()


class _GlobalGenerators:
    """Whose states the global Python and NumPy generators hold: the caller's (holder None) or an engine's.
    Swapping costs more than many engine calls, so with keep set the states are swapped only when another engine
    is called; keep is only for code that draws nothing in between, like the orchestrator loop."""

    def __init__(self):
        self.holder, self.keep, self._callers = None, False, None

    def take(self, engine):
        if self.holder is engine:
            return
        states = random.getstate(), np.random.get_state()
        if self.holder is None:
            self._callers = states
        else:
            self.holder._python_state, self.holder._numpy_state = states
        random.setstate(engine._python_state)
        np.random.set_state(engine._numpy_state)
        self.holder = engine

    def give_back(self):
        """Puts the caller's states back, keeping the holder's for its next call."""
        if self.holder is None:
            return
        self.holder._python_state, self.holder._numpy_state = random.getstate(), np.random.get_state()
        random.setstate(self._callers[0])
        np.random.set_state(self._callers[1])
        self.holder, self._callers = None, None


generators = _GlobalGenerators()
