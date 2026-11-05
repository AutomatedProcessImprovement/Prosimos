"""The load balancer: the order a consumer group's members are offered a message in (docs/orchestrator.md,
"Message routing")."""
from random import Random
from typing import Dict, List, Optional, Union


def offer_order(policy: Optional[Dict[str, Union[Dict[str, float], List[str]]]], candidates: List[str],
                rng: Random) -> List[str]:
    """The order to offer candidates a message in, under their group's policy: None (random), {"weighted":
    {member: weight}} or {"preferred": [members]}. Draws come from rng, the orchestrator's own generator."""
    if policy is None:
        order = list(candidates)
        rng.shuffle(order)
        return order
    [(name, spec)] = policy.items()
    if name == "preferred":
        return [member for member in spec if member in candidates]
    return _weighted(spec, candidates, rng)


def _weighted(weights, candidates, rng):
    # a weight says how often a member is asked first: each draw is among those not asked yet
    left = [member for member in candidates if weights[member] > 0]
    order = []
    while left:
        [member] = rng.choices(left, [weights[member] for member in left])
        order.append(member)
        left.remove(member)
    never_first = [member for member in candidates if weights[member] == 0]
    rng.shuffle(never_first)
    return order + never_first
