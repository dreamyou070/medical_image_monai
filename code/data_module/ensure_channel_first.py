from __future__ import annotations
from collections.abc import Hashable, Mapping
import torch
from monai.config import KeysCollection
from monai.data.meta_tensor import MetaTensor
from monai.transforms.transform import MapTransform
from monai.transforms.utility.array import EnsureChannelFirst
from monai.utils.enums import PostFix

__all__ = ["EnsureChannelFirstD","EnsureChannelFirstDict","EnsureChannelFirstd",]

DEFAULT_POST_FIX = PostFix.meta()

class EnsureChannelFirstd(MapTransform):

    backend = EnsureChannelFirst.backend
    """ dictionary type of data_module (not sure what it does) ... """
    def __init__(self, keys: KeysCollection,
                 strict_check: bool = True,
                 allow_missing_keys: bool = False,
                 channel_dim=None) -> None:

        super().__init__(keys, allow_missing_keys)
        self.adjuster = EnsureChannelFirst(strict_check=strict_check, # True
                                           channel_dim=channel_dim)   # None

    def __call__(self,
                 data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            meta_dict = d[key].meta if isinstance(d[key], MetaTensor) else None  # type: ignore[attr-defined]
            d[key] = self.adjuster(d[key], meta_dict)
        return d


EnsureChannelFirstD = EnsureChannelFirstDict = EnsureChannelFirstd

