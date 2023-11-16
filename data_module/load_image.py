from __future__ import annotations
import numpy as np
from monai.config import DtypeLike, KeysCollection
from monai.data.image_reader import ImageReader
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import PostFix
from abc import ABC, abstractmethod
from monai.utils.enums import TransformBackends
from typing import Any, TypeVar
from monai import config, transforms
from collections.abc import Callable, Generator, Hashable, Iterable, Mapping
from monai.data.meta_tensor import MetaTensor
from utils.check import first

__all__ = ["LoadImaged", "LoadImageD", "LoadImageDict"]

DEFAULT_POST_FIX = PostFix.meta()

class Transform(ABC):

    backend: list[TransformBackends] = []
    @abstractmethod
    def __call__(self, data: Any):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class MapTransform(Transform):

    def __new__(cls, *args, **kwargs):
        if config.USE_META_DICT:
            # call_update after MapTransform.__call__
            cls.__call__ = transforms.attach_hook(cls.__call__, MapTransform.call_update, "post")  # type: ignore

            if hasattr(cls, "inverse"):
                # inverse_update before InvertibleTransform.inverse
                cls.inverse: Any = transforms.attach_hook(cls.inverse, transforms.InvertibleTransform.inverse_update)
        return Transform.__new__(cls)

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__()
        self.keys: tuple[Hashable, ...] = ensure_tuple(keys)
        self.allow_missing_keys = allow_missing_keys
        if not self.keys:
            raise ValueError("keys must be non empty.")
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise TypeError(f"keys must be one of (Hashable, Iterable[Hashable]) but is {type(keys).__name__}.")

    def call_update(self, data):

        if not isinstance(data, (list, tuple, Mapping)):
            return data
        is_dict = False
        if isinstance(data, Mapping):
            data, is_dict = [data], True
        if not data or not isinstance(data[0], Mapping):
            return data[0] if is_dict else data
        list_d = [dict(x) for x in data]  # list of dict for crop samples
        for idx, dict_i in enumerate(list_d):
            for k in dict_i:
                if not isinstance(dict_i[k], MetaTensor):
                    continue
                list_d[idx] = transforms.sync_meta_info(k, dict_i, t=not isinstance(self, transforms.InvertD))
        return list_d[0] if is_dict else list_d

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def key_iterator(self, data: Mapping[Hashable, Any], *extra_iterables: Iterable | None) -> Generator:
        # if no extra iterables given, create a dummy list of Nones
        ex_iters = extra_iterables or [[None] * len(self.keys)]

        # loop over keys and any extra iterables
        _ex_iters: list[Any]
        for key, *_ex_iters in zip(self.keys, *ex_iters):
            # all normal, yield (what we yield depends on whether extra iterables were given)
            if key in data:
                yield (key,) + tuple(_ex_iters) if extra_iterables else key
            elif not self.allow_missing_keys:
                raise KeyError(
                    f"Key `{key}` of transform `{self.__class__.__name__}` was missing in the data_module"
                    " and allow_missing_keys==False."
                )

    def first_key(self, data: dict[Hashable, Any]):
        return first(self.key_iterator(data), ())


class LoadImaged(MapTransform):

    def __init__(self,
                 keys: KeysCollection,
                 reader: type[ImageReader] | str | None = None,
                 dtype: DtypeLike = np.float32,
                 meta_keys: KeysCollection | None = None,
                 meta_key_postfix: str = DEFAULT_POST_FIX,
                 overwriting: bool = False,
                 image_only: bool = True,
                 ensure_channel_first: bool = False,
                 simple_keys: bool = False,
                 prune_meta_pattern: str | None = None,
                 prune_meta_sep: str = ".",
                 allow_missing_keys: bool = False,
                 expanduser: bool = True,
                 *args,
                 **kwargs,
                 ) -> None:

        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader,image_only,
                                 dtype,
                                 ensure_channel_first,
                                 simple_keys,
                                 prune_meta_pattern,
                                 prune_meta_sep,
                                 expanduser,
                                 *args,
                                 **kwargs,)

        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError(f"meta_keys should have the same length as keys, got {len(self.keys)} and {len(self.meta_keys)}.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def register(self, reader: ImageReader):
        self._loader.register(reader)

    def __call__(self,
                 data,
                 reader: ImageReader | None = None):

        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError(
                        f"loader must return a tuple or list (because image_only=False was used), got {type(data)}."
                    )
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError(f"metadata must be a dict, got {type(data[1])}.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        return d

LoadImageD = LoadImageDict = LoadImaged

