from __future__ import annotations
import torch, random
import numpy as np
from typing import Any
from collections.abc import Callable, Iterable, Sequence
import warnings

_flag_deterministic = torch.backends.cudnn.deterministic
_flag_cudnn_benchmark = torch.backends.cudnn.benchmark
NP_MAX = np.iinfo(np.uint32).max
MAX_SEED = NP_MAX + 1


def issequenceiterable(obj: Any) -> bool:
    """ Determine if the object is an iterable sequence and is not a string. """
    try:
        if hasattr(obj, "ndim") and obj.ndim == 0:
            return False  # a 0-d tensor is not iterable
    except Exception:
        return False
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))


def ensure_tuple(vals: Any, wrap_array: bool = False) -> tuple:
    """ vals: input data_module to convert to a tuple.
        if wrap_array = `True`, treat the input numerical array (ndarray/tensor) as one item of the tuple. """
    if wrap_array and isinstance(vals, (np.ndarray, torch.Tensor)):
        return (vals,)
    return tuple(vals) if issequenceiterable(vals) else (vals,)


def set_determinism(seed: int | None = NP_MAX,
                    use_deterministic_algorithms: bool | None = None,
                    additional_settings: Sequence[Callable[[int], Any]] | Callable[[int], Any] | None = None,) -> None:
    """ set seed for deterministic training """
    if seed is None:
        seed_ = torch.default_generator.seed() % MAX_SEED
        torch.manual_seed(seed_)
    else:
        seed = int(seed) % MAX_SEED
        torch.manual_seed(seed)

    global _seed
    _seed = seed
    random.seed(seed)
    np.random.seed(seed)

    if additional_settings is not None:
        additional_settings = ensure_tuple(additional_settings)
        for func in additional_settings:
            func(seed)

    if torch.backends.flags_frozen():
        warnings.warn("PyTorch global flag support of backends is disabled, enable it to set global `cudnn` flags.")
        torch.backends.__allow_nonbracketed_mutation_flag = True

    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # restore the original flags
        torch.backends.cudnn.deterministic = _flag_deterministic
        torch.backends.cudnn.benchmark = _flag_cudnn_benchmark
    if use_deterministic_algorithms is not None:
        if hasattr(torch, "use_deterministic_algorithms"):  # `use_deterministic_algorithms` is new in torch 1.8.0
            torch.use_deterministic_algorithms(use_deterministic_algorithms)
        elif hasattr(torch, "set_deterministic"):  # `set_deterministic` is new in torch 1.7.0
            torch.set_deterministic(use_deterministic_algorithms)
        else:
            warnings.warn("use_deterministic_algorithms=True, but PyTorch version is too old to set the mode.")
