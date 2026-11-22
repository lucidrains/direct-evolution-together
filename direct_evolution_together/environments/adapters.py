"""Server-side simulator adapters keyed by task id."""

from __future__ import annotations

from .cartpole import CartPoleAdapter
from .halfcheetah import HalfCheetahAdapter
from .lunarlander import LunarLanderAdapter
from .registry import get_task_spec
from .types import ControlTaskAdapter

TASK_ADAPTERS: dict[str, ControlTaskAdapter] = {
    "cartpole": CartPoleAdapter(),
    "halfcheetah": HalfCheetahAdapter(),
    "lunarlander": LunarLanderAdapter(),
}


def get_task_adapter(task_id: str) -> ControlTaskAdapter:
    spec = get_task_spec(task_id)
    adapter = TASK_ADAPTERS.get(spec.task_id)
    if adapter is None:
        raise ValueError(f"Task '{task_id}' does not support server control simulation.")
    return adapter
