from __future__ import annotations
import numpy as np
from monai.config import DtypeLike, KeysCollection
from monai.data.image_reader import ImageReader
from monai.transforms.io.array import LoadImage, SaveImage
from monai.transforms.transform import MapTransform, Transform
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import PostFix

__all__ = ["LoadImaged", "LoadImageD", "LoadImageDict"]

DEFAULT_POST_FIX = PostFix.meta()


class LoadImaged(MapTransform):

    def __init__(
        self,
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
            raise ValueError(
                f"meta_keys should have the same length as keys, got {len(self.keys)} and {len(self.meta_keys)}."
            )
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def register(self, reader: ImageReader):
        self._loader.register(reader)

    def __call__(self, data, reader: ImageReader | None = None):

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

