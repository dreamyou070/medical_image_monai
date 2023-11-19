from __future__ import annotations
from monai.transforms.traits import LazyTrait
from collections.abc import Callable
from typing import Any, TypeVar
from data_module.data_preprocessing import data_info_preprocess

ReturnType = TypeVar("ReturnType")
__override_keywords = {"mode", "padding_mode", "dtype", "align_corners", "resample_mode", "device"}

def apply_transform(transform: Callable[..., ReturnType],
                    data: Any,
                    unpack_parameters: bool = False,
                    lazy: bool | None = False,
                    overrides: dict | None = None,
                    logger_name: bool | str = False,) -> ReturnType:

    data = data_info_preprocess(transform, data, lazy, overrides, logger_name)

    if isinstance(data, tuple) and unpack_parameters:
        return transform(*data, lazy=lazy) if isinstance(transform, LazyTrait) else transform(*data)
    return transform(data, lazy=lazy) if isinstance(transform, LazyTrait) else transform(data)

#transforms.RandAffined