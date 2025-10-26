from __future__ import annotations

import contextvars
import os

# Feature flag for using compiled evaluator instead of per-node generators, managed like caching switches
_env_compiled_flag = os.getenv("KRROOD_USE_COMPILED_EVALUATOR", "0") not in (
    "0",
    "false",
    "False",
    None,
)
_compiled_evaluator_enabled = contextvars.ContextVar(
    "compiled_evaluator_enabled", default=_env_compiled_flag
)


def enable_compiled_evaluator() -> None:
    """
    Enable the compiled evaluator fast-paths.
    """
    _compiled_evaluator_enabled.set(True)


def disable_compiled_evaluator() -> None:
    """
    Disable the compiled evaluator fast-paths.
    """
    _compiled_evaluator_enabled.set(False)


def is_compiled_evaluator_enabled() -> bool:
    """
    Check whether the compiled evaluator is enabled.
    """
    return _compiled_evaluator_enabled.get()


def use_compiled_evaluator(enabled: bool = True) -> None:
    """
    Backward-compatible toggle for the compiled evaluator.

    When enabled, ResultQuantifier.evaluate() may execute a single evaluator
    compiled from the whole query instead of per-node evaluation. This keeps
    public APIs intact while improving performance.
    """
    if enabled:
        enable_compiled_evaluator()
    else:
        disable_compiled_evaluator()
