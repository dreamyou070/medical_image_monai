from __future__ import annotations
import collections.abc
from collections.abc import Callable, Sequence
from typing import IO, TYPE_CHECKING, Any
import numpy as np
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset
from data_module.transforms import apply_transform
from data_module.load_image import LoadImaged
from data_module.ensure_channel_first import EnsureChannelFirstd
from data_module.scaling import ScaleIntensityRanged
from data_module.affine_transferling import RandAffined
import os
from monai.utils import min_version, optional_import
from monai import transforms
import warnings
import torch
from torch.utils.data import DataLoader as _TorchDataLoader
from torch.utils.data import Dataset
from monai.data.meta_obj import get_track_meta
from monai.data.utils import list_data_collate, set_rnd, worker_init_fn
from PIL import Image
if TYPE_CHECKING:
    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")

cp, _ = optional_import("cupy")
lmdb, _ = optional_import("lmdb")
pd, _ = optional_import("pandas")
kvikio_numpy, _ = optional_import("kvikio.numpy")


__all__ = ["SYDataset", "SYDataLoader"]



def get_transform(image_size):
    w,h = image_size.split(',')
    img_loader = LoadImaged(keys=["image"])
    channel_orderer = EnsureChannelFirstd(keys=["image"])
    image_range_changer = ScaleIntensityRanged(keys=["image"],
                                  a_min=0.0, a_max=255.0, # original pixel range
                                  b_min=0.0, b_max=1.0,   # target pixel range
                                  clip=True)
    # spatial_size = output image spatial size.
    #                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1, the transform will use the spatial size of `img`.
    # 2) prob
    affine_transformer = RandAffined(keys=["image"],
                                     spatial_size=[int(w.strip()), int(h.strip())],
                                     prob=0.5,
                                     rotate_range=[(-np.pi / 36, np.pi / 36),(-np.pi / 36, np.pi / 36)],
                                     translate_range=[(-1, 1), (-1, 1)],
                                     scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
                                     padding_mode="zeros",)
    train_transforms = transforms.Compose([img_loader,
                                           channel_orderer,
                                           image_range_changer,
                                           affine_transformer])

    val_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                         transforms.EnsureChannelFirstd(keys=["image"]),
                                         transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0,
                                                                         b_min=0.0, b_max=1.0, clip=True), ])
    return train_transforms, val_transforms

def val_transform():

    val_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                         transforms.EnsureChannelFirstd(keys=["image"]),
                                         transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0,
                                                                         b_min=0.0, b_max=1.0, clip=True), ])
    return val_transforms

class SYDataset_masking(_TorchDataset):

    def __init__(self,
                 data: Sequence,
                 transform: Callable | None = None,
                 base_mask_dir : str = None,
                 image_size = 64,) -> None:

        self.data = data # list of datas
        self.transform: Any = transform
        self.reverse_indexing = True
        self.image_size = image_size
        self.base_mask_dir = base_mask_dir

    def __len__(self) -> int:
        return len(self.data)

    def data_transform(self, index: int):

        data_i = self.data[index]
        preprocessed_data = apply_transform(self.transform, data_i)
        return preprocessed_data

    def __getitem__(self, index: int | slice | Sequence[int]):

        data_dict = {}

        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)

        if isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)

        data_dir = self.data[index]['image']
        parent, net_name = os.path.split(data_dir)
        mask_dir = os.path.join(self.base_mask_dir, net_name)
        mask_pil = Image.open(mask_dir)
        mask_np = np.array(mask_pil)
        criterion = np.sum(mask_np)
        normal = True
        if criterion > 0 :
            normal = False
        w,h = mask_pil.size
        resized_masK_pil = mask_pil.resize((int(w/4) - 2, int(h/4) - 2))
        resized_masK_np = np.array(resized_masK_pil)
        data_dict['image_info'] = self.data_transform(index)
        data_dict['normal'] = int(normal)
        data_dict['mask'] = 1-torch.from_numpy(resized_masK_np)

        return data_dict

class SYDataset(_TorchDataset):

    def __init__(self,
                 data: Sequence,
                 transform: Callable | None = None,
                 image_size = 64) -> None:

        self.data = data # list of datas
        self.transform: Any = transform
        self.reverse_indexing = True
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.data)

    def data_transform(self, index: int):

        data_i = self.data[index]
        preprocessed_data = apply_transform(self.transform, data_i)
        return preprocessed_data

    def __getitem__(self, index: int | slice | Sequence[int]):

        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)

        if isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)

        return self.data_transform(index)




class SYDataLoader(_TorchDataLoader):

    def __init__(self, dataset: Dataset, num_workers: int = 0, **kwargs) -> None:
        if num_workers == 0:
            # when num_workers > 0, random states are determined by worker_init_fn
            # this is to make the behavior consistent when num_workers == 0
            # torch.int64 doesn't work well on some versions of windows
            _g = torch.random.default_generator if kwargs.get("generator") is None else kwargs["generator"]
            init_seed = _g.initial_seed()
            _seed = torch.empty((), dtype=torch.int64).random_(generator=_g).item()
            set_rnd(dataset, int(_seed))
            _g.manual_seed(init_seed)
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = list_data_collate
        if "worker_init_fn" not in kwargs:
            kwargs["worker_init_fn"] = worker_init_fn

        if (
            "multiprocessing_context" in kwargs
            and kwargs["multiprocessing_context"] == "spawn"
            and not get_track_meta()):
            warnings.warn(
                "Please be aware: Return type of the dataloader will not be a Tensor as expected but"
                " a MetaTensor instead! This is because 'spawn' creates a new process where _TRACK_META"
                " is initialized to True again. Context:_TRACK_META is set to False and"
                " multiprocessing_context to spawn"
            )

        super().__init__(dataset=dataset, num_workers=num_workers, **kwargs)

