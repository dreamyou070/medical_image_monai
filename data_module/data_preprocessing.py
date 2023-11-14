from __future__ import annotations
from monai.transforms.traits import LazyTrait
from typing import Any, TypeVar
import torch
from monai.transforms.lazy.dictionary import ApplyPendingd
from monai.utils import LazyAttr, look_up_option
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import to_affine_nd
from monai.transforms.lazy.utils import (affine_from_pending,combine_transforms,is_compatible_apply_kwargs,kwargs_from_pending,resample,)

ReturnType = TypeVar("ReturnType")
__override_keywords = {"mode", "padding_mode", "dtype", "align_corners", "resample_mode", "device"}

def apply_pending(data,
                  pending: list | None = None, overrides: dict | None = None):

    overrides = (overrides or {}).copy()
    for k in overrides:
        look_up_option(k, __override_keywords)  # check existence of the key

    if isinstance(data, MetaTensor) and pending is None:
        pending = data.pending_operations.copy()
        data.clear_pending_operations()
    pending = [] if pending is None else pending

    if not pending:
        return data, []

    cumulative_xform = affine_from_pending(pending[0])
    if cumulative_xform.shape[0] == 3:
        cumulative_xform = to_affine_nd(3, cumulative_xform)

    cur_kwargs = kwargs_from_pending(pending[0])
    override_kwargs: dict[str, Any] = {}
    if "mode" in overrides:
        override_kwargs[LazyAttr.INTERP_MODE] = overrides["mode"]
    if "padding_mode" in overrides:
        override_kwargs[LazyAttr.PADDING_MODE] = overrides["padding_mode"]
    if "align_corners" in overrides:
        override_kwargs[LazyAttr.ALIGN_CORNERS] = overrides["align_corners"]
    if "resample_mode" in overrides:
        override_kwargs[LazyAttr.RESAMPLE_MODE] = overrides["resample_mode"]
    override_dtype = overrides.get("dtype", torch.float64)
    override_kwargs[LazyAttr.DTYPE] = data.dtype if override_dtype is None else override_dtype
    device = overrides.get("device")

    for p in pending[1:]:
        new_kwargs = kwargs_from_pending(p)
        if not is_compatible_apply_kwargs(cur_kwargs, new_kwargs):
            # carry out an intermediate resample here due to incompatibility between arguments
            _cur_kwargs = cur_kwargs.copy()
            _cur_kwargs.update(override_kwargs)
            data = resample(data.to(device), cumulative_xform, _cur_kwargs)

        next_matrix = affine_from_pending(p)
        if next_matrix.shape[0] == 3:
            next_matrix = to_affine_nd(3, next_matrix)

        cumulative_xform = combine_transforms(cumulative_xform, next_matrix)
        cur_kwargs.update(new_kwargs)
    cur_kwargs.update(override_kwargs)
    data = resample(data.to(device), cumulative_xform, cur_kwargs)
    if isinstance(data, MetaTensor):
        for p in pending:
            data.push_applied_operation(p)
    return data, pending


def apply_transforms(data,
                     keys: tuple | None,
                     overrides: dict | None = None,
                     logger_name: bool | str = False,):

    if isinstance(data, list):
        return [apply_transforms(d, keys, overrides, logger_name) for d in data]

    if isinstance(data, tuple):
        return tuple(apply_transforms(d, keys, overrides, logger_name) for d in data)

    if isinstance(data, dict):

        # ---------------------------------------------------------------------------------------------------------------
        # 1) active keys
        active_keys = [k for k in data.keys() if keys is None or k in keys]

        # ---------------------------------------------------------------------------------------------------------------
        # 2) keys_to_update
        keys_to_update = [k for k in active_keys if isinstance(data[k], MetaTensor) and data[k].has_pending_operations]
        if len(keys_to_update) > 0:
            rdata = dict(data)
            for k in keys_to_update:
                overrides_ = None if overrides is None else overrides.get(k, None)
                rdata[k], _ = apply_pending(data[k], overrides=overrides_)
            return rdata

    else:
        if isinstance(data, MetaTensor) and data.has_pending_operations:
            rdata, _ = apply_pending(data, overrides=overrides)
            return rdata
    return data

def data_info_preprocess(transform,
                                data,
                                lazy: bool | None = None,
                                overrides: dict | None = None,
                                logger_name: bool | str = False):

    must_apply_pending = True
    keys = transform.keys if isinstance(transform, ApplyPendingd) else None # None
    if isinstance(transform, LazyTrait) and not transform.requires_current_data:
        must_apply_pending = not (transform.lazy if lazy is None else lazy)
    if must_apply_pending is True:
        """ must apply pending transforms """
        return apply_transforms(data, keys, overrides, logger_name)
    return data
