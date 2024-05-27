import torch

from mvsdf.data.utils import center_crop_tensor, get_center_crop


class FixedEdgesAndIntegerPixelCentersInCalibration:
    r"""In this model we make two assumptions:
    1. The edges of images are fixed during rescaling.
    2. The cameras are calibrated so that
    UV coordinates of the center of the upper left pixel in the camera coordinate system are 0, 0
    and the coordinates of the center of the lower right pixel are width - 1, height - 1.

    For the first assumption to hold we need to use `align_corners=False`
    in all image sampling/resampling PyTorch methods.
    For the second assumption to hold we need the appropriate calibration.
    We rely on DTU dataset, which is calibrated using this toolbox http://www.vision.caltech.edu/bouguetj/calib_doc/ ---
    see "Important Convention" there at http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html.
    """

    @staticmethod
    def interpolate(*args, recompute_scale_factor=False, **kwargs):
        r"""Implements torch.nn.functional.interpolate.

        `recompute_scale_factor` is False by default since 1.6.0
        """
        align_corners = False
        if 'mode' in kwargs and kwargs['mode'] in {'nearest', 'area'}:
            align_corners = None  # these modes always function as with align_corners=False
        return torch.nn.functional.interpolate(*args, align_corners=align_corners,
                                               recompute_scale_factor=recompute_scale_factor, **kwargs)

    @staticmethod
    def interpolate_depth(depth, *args, **kwargs):
        r"""Implements torch.nn.functional.interpolate for depth maps."""
        # TODO maybe change to hole aware box resampling
        return FixedEdgesAndIntegerPixelCentersInCalibration.interpolate(depth, *args, mode='nearest', **kwargs)

    @staticmethod
    def rescale_intrinsics(intrinsics, scaling_factor, inplace=False):
        r"""Rescale intrinsics so that ``(c_x + 1/2) / width = const`` and ``f_x / width = const``.

        Parameters
        ----------
        intrinsics : torch.Tensor
            of shape [**, 3, 3]
        scaling_factor : float
        inplace : bool

        Returns
        -------
        intrinsics : torch.Tensor
            of shape [**, 3, 3]
        """
        if not inplace:
            intrinsics = intrinsics.clone()
        intrinsics[..., 0, 0] *= scaling_factor
        intrinsics[..., 1, 1] *= scaling_factor
        intrinsics[..., :2, 2] = (intrinsics[..., :2, 2] + .5) * scaling_factor - .5
        return intrinsics

    @staticmethod
    def center_crop_tensor(tensor, new_size):
        r"""

        Parameters
        ----------
        tensor : torch.Tensor
            of shape [..., height, width]
        new_size : iterable of int
            [new_height, new_width]

        Returns
        -------
        tensor : torch.Tensor
            of shape [..., new_height, new_width]
        """
        return center_crop_tensor(tensor, new_size)

    @staticmethod
    def center_crop_intrinsics(intrinsics, old_size, new_size, inplace=False):
        r"""

        Parameters
        ----------
        intrinsics : torch.Tensor
            of shape [**, 3, 3]
        old_size : iterable of int
            [height, width]
        new_size : iterable of int
            [new_height, new_width]
        inplace : bool

        Returns
        -------
        intrinsics : torch.Tensor
            of shape [**, 3, 3]
        """
        if not inplace:
            intrinsics = intrinsics.clone()
        crop_start, _ = get_center_crop(old_size, new_size)
        intrinsics[..., 0, 2] -= crop_start[1]
        intrinsics[..., 1, 2] -= crop_start[0]
        return intrinsics

    @staticmethod
    def grid_sample(*args, **kwargs):
        r"""Implements torch.nn.functional.grid_sample"""
        return torch.nn.functional.grid_sample(*args, align_corners=False, **kwargs)

    @staticmethod
    def make_uv_grid(size, dtype, device):
        r"""Implements torch.meshgrid(linspace(height), linspace(width))
        with u in [0, width - 1] and v in [0, height - 1].

        Parameters
        ----------
        size : tuple
            [height, width]
        dtype
        device

        Returns
        -------
        u : torch.Tensor
        v : torch.Tensor
        """
        height, width = size
        v, u = torch.meshgrid([torch.linspace(0, height - 1, height, dtype=dtype, device=device),
                               torch.linspace(0, width - 1, width, dtype=dtype, device=device)])
        return u, v

    @staticmethod
    def uv_to_pixel_id(uv, dtype=None):
        r"""
        Parameters
        ----------
        uv : torch.Tensor
        dtype : torch.dtype

        Returns
        -------
        ji : torch.Tensor
        """
        if dtype is None:
            dtype = uv.dtype
        return torch.empty_like(uv, dtype=dtype).copy_(uv)

    @staticmethod
    def pixel_id_to_uv(ji, dtype=None):
        r"""
        Parameters
        ----------
        ji : torch.Tensor
        dtype : torch.dtype

        Returns
        -------
        uv : torch.Tensor
        """
        if dtype is None:
            dtype = ji.dtype
        return torch.empty_like(ji, dtype=dtype).copy_(ji)

    @staticmethod
    def normalize_uv(u_or_v, size):
        r"""Rescales tensor of u or v coordinates with the respective size=width or =height to [-1, 1],
        so that -0.5 -> -1 and (size - 0.5) -> 1.

        Parameters
        ----------
        u_or_v : torch.Tensor
        size : float
        """
        return u_or_v / (size / 2) - (size - 1) / size
