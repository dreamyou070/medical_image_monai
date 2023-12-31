from __future__ import annotations
from collections.abc import Hashable, Mapping, Sequence
import numpy as np
import torch
from monai.config import DtypeLike, KeysCollection, SequenceStr
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.spatial.array import RandAffine
from monai.transforms.transform import LazyTransform, MapTransform, RandomizableTransform
from monai.utils import (GridSampleMode,GridSamplePadMode,convert_to_tensor,ensure_tuple,
                         ensure_tuple_rep,fall_back_tuple,)
from monai.utils.enums import TraceKeys
from monai.utils.module import optional_import

nib, _ = optional_import("nibabel")

__all__ = ["RandAffineD","RandAffineDict",]


class RandAffined(RandomizableTransform, MapTransform, InvertibleTransform, LazyTransform):

    backend = RandAffine.backend

    def __init__(self,
                 keys: KeysCollection,
                 spatial_size: Sequence[int] | int | None = None,
                 prob: float = 0.1,
                 rotate_range: Sequence[tuple[float, float] | float] | float | None = None,
                 shear_range: Sequence[tuple[float, float] | float] | float | None = None,
                 translate_range: Sequence[tuple[float, float] | float] | float | None = None,
                 scale_range: Sequence[tuple[float, float] | float] | float | None = None,
                 mode: SequenceStr = GridSampleMode.BILINEAR,
                 padding_mode: SequenceStr = GridSamplePadMode.REFLECTION,
                 cache_grid: bool = False,
                 device: torch.device | None = None,
                 allow_missing_keys: bool = False,
                 lazy: bool = False,) -> None:
        """
        Args:
            spatial_size: output image spatial size.
            rotate_range: angle range in radians.

            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D, a tuple of 6 floats for 3D) for affine matrix,
                take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select pixel/voxel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                It also can be a sequence, each element corresponds to a key in ``keys``.
            cache_grid: whether to cache the identity sampling grid.
                If the spatial size is not dynamically defined by input image, enabling this option could
                accelerate the transform.
            device: device on which the tensor will be allocated.
            allow_missing_keys: don't raise exception if key is missing.
            lazy: a flag to indicate whether this transform should execute lazily or not.
                Defaults to False

        See also:
            - :py:class:`monai.transforms.compose.MapTransform`
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        LazyTransform.__init__(self, lazy=lazy)
        self.rand_affine = RandAffine(
            prob=1.0,  # because probability handled in this class
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            spatial_size=spatial_size,
            cache_grid=cache_grid,
            device=device,
            lazy=lazy,
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        self._lazy = val
        self.rand_affine.lazy = val

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandAffined:
        self.rand_affine.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor], lazy: bool | None = None
    ) -> dict[Hashable, NdarrayOrTensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data_module to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data_module, as well as any other data_module present in the dictionary
        """
        d = dict(data)
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            out: dict[Hashable, NdarrayOrTensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out

        self.randomize(None)
        # all the keys share the same random Affine factor
        self.rand_affine.randomize()

        item = d[first_key]
        spatial_size = item.peek_pending_shape() if isinstance(item, MetaTensor) else item.shape[1:]
        lazy_ = self.lazy if lazy is None else lazy

        sp_size = fall_back_tuple(self.rand_affine.spatial_size, spatial_size)
        # change image size or do random transform
        do_resampling = self._do_transform or (sp_size != ensure_tuple(spatial_size))
        # converting affine to tensor because the resampler currently only support torch backend
        grid = None
        if do_resampling:  # need to prepare grid
            grid = self.rand_affine.get_identity_grid(sp_size, lazy=lazy_)
            if self._do_transform:  # add some random factors
                grid = self.rand_affine.rand_affine_grid(sp_size, grid=grid, lazy=lazy_)
        grid = 0 if grid is None else grid  # always provide a grid to self.rand_affine

        for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
            # do the transform
            if do_resampling:
                d[key] = self.rand_affine(d[key], None, mode, padding_mode, True, grid, lazy=lazy_)  # type: ignore
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta(), dtype=torch.float32)
            self._do_transform = do_resampling  # TODO: unify self._do_transform and do_resampling
            self.push_transform(d[key], replace=True, lazy=lazy_)
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            tr = self.pop_transform(d[key])
            if TraceKeys.EXTRA_INFO not in tr[TraceKeys.EXTRA_INFO]:
                continue
            do_resampling = tr[TraceKeys.EXTRA_INFO][TraceKeys.EXTRA_INFO]["do_resampling"]
            if do_resampling:
                d[key].applied_operations.append(tr[TraceKeys.EXTRA_INFO])  # type: ignore
                d[key] = self.rand_affine.inverse(d[key])  # type: ignore

        return d

RandAffineD = RandAffineDict = RandAffined
