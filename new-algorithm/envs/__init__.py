"""DeepRacer environments package.

Env classes are exported lazily (PEP 562): importing a genesis-free leaf like
``envs.features`` or ``envs.mdp`` must not pull in the env classes, whose
import chain loads genesis — that would poison genesis-free processes (the
ONNX exporter, which cannot coexist with onnxruntime's LLVM symbols, and any
spec ``validate()`` call that only needs tracks/features/signals).
"""

_LAZY = {
    "DeepRacerEnv": ".deepracer_env",
    "VectorDeepRacerEnv": ".deepracer_env",
    "VisionDeepRacerEnv": ".deepracer_env",
    "Track": ".track",
    "TRACKS": ".track",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
