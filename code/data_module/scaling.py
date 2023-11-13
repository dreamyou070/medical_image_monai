from __future__ import annotations
from typing import Hashable, Mapping
from monai.config import KeysCollection
from monai.transforms.intensity.array import ScaleIntensityRange
from monai.transforms.transform import MapTransform
from monai.utils.enums import PostFix
from warnings import warn
import numpy as np
from monai.config import DtypeLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms.transform import RandomizableTransform, Transform
from monai.transforms.utils_pytorch_numpy_unification import clip
from monai.utils.enums import TransformBackends
from monai.utils.module import min_version, optional_import
from monai.utils.type_conversion import convert_data_type, convert_to_tensor

skimage, _ = optional_import("skimage", "0.19.0", min_version)
__all__ = ["ScaleIntensityRangeD","ScaleIntensityRangeDict","ScaleIntensityRange"]

DEFAULT_POST_FIX = PostFix.meta()

class ScaleIntensityRange(Transform):
    """
    if clip=True is about clipping
    If `clip=True`, when `b_min`/`b_max` is None, the clipping is not performed on the corresponding edge.
    Args:
        a : intensity original range.
        b : intensity target range.
        clip: whether to perform clip after scaling.
        dtype: output data_module type, if None, same as input image. defaults to float32.
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    def __init__(self,a_min: float,a_max: float,
                 b_min: float | None = None,b_max: float | None = None,
                 clip: bool = False,dtype: DtypeLike = np.float32,) -> None:
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.dtype = dtype

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """Apply the transform to `img`."""
        img = convert_to_tensor(img, track_meta=get_track_meta())
        dtype = self.dtype or img.dtype
        if self.a_max - self.a_min == 0.0:
            warn("Divide by zero (a_min == a_max)", Warning)
            if self.b_min is None:
                return img - self.a_min
            return img - self.a_min + self.b_min

        img = (img - self.a_min) / (self.a_max - self.a_min)
        if (self.b_min is not None) and (self.b_max is not None):
            img = img * (self.b_max - self.b_min) + self.b_min
        if self.clip:
            img = clip(img, self.b_min, self.b_max)
        ret: NdarrayOrTensor = convert_data_type(img, dtype=dtype)[0]

        return ret


class ScaleIntensityRanged(MapTransform):

    backend = ScaleIntensityRange.backend

    def __init__(
        self,
        keys: KeysCollection,
        a_min: float,
        a_max: float,
        b_min: float | None = None,
        b_max: float | None = None,
        clip: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ScaleIntensityRange(a_min, a_max, b_min, b_max, clip, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d


ScaleIntensityRangeD = ScaleIntensityRangeDict = ScaleIntensityRanged
