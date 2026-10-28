"""
The random generators' starting states: each engine, pick and catalogue draws from generators of its own, seeded
from the run's seed and its name, so that one doesn't change the others' draws.
"""
import hashlib
import random

import numpy as np


def _engine_random_states(seed, process_name):
    """Initial Python and NumPy generator states for one engine. A stable hash of the seed and the
    process name, not Python's hash(), which differs between runs."""
    if seed is None:
        python_seed, numpy_seed = random.getrandbits(64), random.getrandbits(32)
    else:
        digest = hashlib.sha256(f"{seed}/{process_name}".encode("utf-8")).digest()
        python_seed, numpy_seed = int.from_bytes(digest[:8], "big"), int.from_bytes(digest[8:12], "big")
    return random.Random(python_seed).getstate(), np.random.RandomState(numpy_seed).get_state()
