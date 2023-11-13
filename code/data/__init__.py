from __future__ import annotations
import collections.abc
from collections.abc import Callable, Sequence
from typing import IO, TYPE_CHECKING, Any
import numpy as np
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset
from monai.transforms import Compose ,Randomizable ,RandomizableTrait ,Transform ,apply_transform
from monai.utils import MAX_SEED, convert_to_tensor, get_seed, look_up_option, min_version, optional_import
from monai import transforms
import warnings
import torch
from torch.utils.data import DataLoader as _TorchDataLoader
from torch.utils.data import Dataset
from monai.data.meta_obj import get_track_meta
from monai.data.utils import list_data_collate, set_rnd, worker_init_fn
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
    train_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                           transforms.EnsureChannelFirstd(keys=["image"]),
                                           transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0,
                                                                           b_min=0.0, b_max=1.0, clip=True),
                                           transforms.RandAffined(keys=["image"],
                                                                  rotate_range=[(-np.pi / 36, np.pi / 36),
                                                                                (-np.pi / 36, np.pi / 36)],
                                                                  translate_range=[(-1, 1), (-1, 1)],
                                                                  scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
                                                                  spatial_size=[image_size, image_size],
                                                                  padding_mode="zeros",
                                                                  prob=0.5, ), ])
    val_transforms = transforms.Compose([transforms.LoadImaged(keys=["image"]),
                                         transforms.EnsureChannelFirstd(keys=["image"]),
                                         transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0,
                                                                         b_min=0.0, b_max=1.0, clip=True), ])
    return train_transforms, val_transforms


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

    def _transform(self, index: int):
        """read and transforming the data. """
        data_i = self.data[index]
        preprocessed_data = apply_transform(self.transform, data_i) if self.transform is not None else data_i
        return preprocessed_data

    def __getitem__(self, index: int | slice | Sequence[int]):

        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)

        if isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)

        return self._transform(index)




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

