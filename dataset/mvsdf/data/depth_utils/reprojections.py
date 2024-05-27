import torch

from mvsdf.data.image_coordinates_model.default import UVModel
from mvsdf.utils.matrix_inverse import kludgy_matrix_inverse


def depth_to_absolute_coordinates(depth, depth_type, intrinsic_matrix, stereo_baseline=None,
                                  image_coordinates_model=UVModel):
    """Reconstructs 3d coordinates of depth map pixels.

    Parameters
    ----------
    depth : torch.Tensor
        Depth map of shape [**, height, width].
    depth_type : {'perspective', 'orthogonal', 'disparity'}
        Type of the depth map: 'perspective' denotes the distance from point to camera,
        'orthogonal' denotes the distance from point to image plane.
    intrinsic_matrix : torch.Tensor
        of shape [**, 3, 3].
    stereo_baseline : torch.Tensor or float
        Tensor of shape [**] or broadcastable or single float. Required for 'disparity' depth_type.
    image_coordinates_model

    Returns
    -------
    coordinates : torch.Tensor
        Coordinates of the points [**, 3, height, width] in the camera coordinate system:
        X points to the right, Y points down, Z points forward.
    """
    dtype = depth.dtype
    device = depth.device
    height, width = depth.shape[-2:]

    u, v = image_coordinates_model.make_uv_grid([height, width], dtype=dtype, device=device)
    u, v = u.reshape(-1), v.reshape(-1)
    points = torch.stack([u, v, torch.ones_like(u)]);  del u, v
    points = torch.inverse(intrinsic_matrix) @ points
    # points = kludgy_matrix_inverse(intrinsic_matrix) @ points
    points = points.view(*intrinsic_matrix.shape[:-1], height, width)
    if depth_type == 'orthogonal':
        points = points / points[..., 2:3, :, :]
        return points * depth.unsqueeze(-3)
    elif depth_type == 'perspective':
        points = torch.nn.functional.normalize(points, dim=-3)
        return points * depth.unsqueeze(-3)
    elif depth_type == 'disparity':
        points = points / points[..., 2:3, :, :]
        stereo_baseline = torch.as_tensor(stereo_baseline).to(intrinsic_matrix)
        focal_length = intrinsic_matrix[..., 0, 0]
        depth = (stereo_baseline * focal_length).unsqueeze(-1).unsqueeze(-1) / depth
        return points * depth.unsqueeze(-3)


def recompute_depth(src_depth, src_intrinsics, src_extrinsics, ref_extrinsics):
    r"""Calculates z-depths of the source pixels w.r.t the reference camera,
    given the source camera intrinsic and extrinsic matrices, the reference camera extrinsic matrix,
    and z-depths of the source pixels w.r.t the source camera.

    I.e, resulting values are calculated as
        z_in_src -> xyz_in_src -> xyz_in_world -> xyz_in_ref -> z_in_ref.

    Parameters
    ----------
    src_depth : torch.Tensor
        of shape [batch_size, height, width]
    src_intrinsics : torch.Tensor
        of shape [batch_size, 3, 3]
    src_extrinsics : torch.Tensor
        of shape [batch_size, 4, 4]
    ref_extrinsics : torch.Tensor
        of shape [batch_size, 4, 4]

    Returns
    -------
    src_pixel_z_in_ref : torch.Tensor
        of shape [batch_size, height, width]
    """
    batch_size, height, width = src_depth.shape

    xyz_src = depth_to_absolute_coordinates(src_depth, 'orthogonal', src_intrinsics); del src_depth
    xyz1_src = torch.cat([xyz_src, xyz_src.new_ones(batch_size, 1, height, width)], 1).view(batch_size, 4, -1)
    del xyz_src
    from_src_to_ref = ref_extrinsics @ src_extrinsics.inverse()
    from_src_to_ref_z = from_src_to_ref; del from_src_to_ref
    z_ref_recomputed = (from_src_to_ref_z @ xyz1_src); del from_src_to_ref_z, xyz1_src
    
    return z_ref_recomputed[..., 2, :].view(batch_size, height, width)


def ref_depth_to_src_uv(sampled_tensor, src_camera_matrix, ref_camera_matrix, ref_depth, div_eps=1e-7, normalize=True):
    r"""Calculates UV coordinates of the reference pixels in the coordinate system of the source camera,
    given the source and reference camera matrices, and z-depths of the reference pixels (w.r.t reference camera).

    I.e, the resulting values are calculated as
        (uv_in_ref, ref_depth) -> xyz_in_ref -> xyz_in_world -> xyz_in_src -> uv_in_src

    Parameters
    ----------
    sampled_tensor : torch.Tensor
        of shape [batch_size, channels_n, height, width].
        Only used to extract dtype, device and output shape, e.g can be an empty tensor.
    src_camera_matrix : torch.Tensor
        of shape [batch_size, 4, 4]
    ref_camera_matrix : torch.Tensor
        of shape [batch_size, 4, 4]
    ref_depth : torch.Tensor
        of shape [batch_size, depths_batch_size, height, width] or broadcastable
    div_eps : float
        Small value to avoid division by zero. In AMP don't use values less than 1e-7.
    normalize : bool
        If True, the UV coordinates are normalized to [-1, 1] (for grid_sample).

    Returns
    -------
    uv : torch.Tensor
        of shape [batch_size, depths_batch_size, height, width, 2]

    Warnings
    --------
    This function is non-differentiable by design. If you want a differentiable version, use something like this
    https://github.com/voyleg/dev.mvs4df/blob/131da7c9f6ddbd190f9ab578b684818f0d336793/src/mvs4df/data/depth_utils/reprojections.py#L89

    Notes
    -----
    Here we assume that cameras are calibrated so that
    UV coordinates of the center of the upper left pixel in the camera coordinate system are 0, 0
    and the coordinates of the center of the lower right pixel are width - 1, height - 1.
    """
    batch_size, channels_n, height, width = sampled_tensor.shape
    depths_batch_size = ref_depth.shape[1]
    dtype, device = sampled_tensor.dtype, sampled_tensor.device

    with torch.no_grad():
        inv_reference_camera_matrix = kludgy_matrix_inverse(ref_camera_matrix)
        with torch.cuda.amp.autocast(enabled=False):  # in AMP there's some significant loss of precision here
            if (src_camera_matrix.dtype is torch.float16) or (inv_reference_camera_matrix.dtype is torch.float16):
                _ = src_camera_matrix.float() @ inv_reference_camera_matrix.float()
            else:  # assuming double (or float)
                _ = src_camera_matrix @ inv_reference_camera_matrix
        R = _[:, :3, :3]
        t = _[:, :3, -1]

        u, v = UVModel.make_uv_grid([height, width], dtype=dtype, device=device)
        u, v = u.contiguous().view(-1), v.contiguous().view(-1)
        uvo = torch.stack([u, v, torch.ones_like(u)])  # 3, h * w
        del u, v

        uvo = uvo.view(1, 3, -1)
        t = t.view(batch_size, 3, 1, 1)
        ref_depth = ref_depth.view(batch_size, 1, depths_batch_size, -1)
        _ = (R @ uvo).view(batch_size, 3, 1, -1) + t / ref_depth.clamp(min=div_eps)  # b, 3, d, h * w
        del R, uvo, t, ref_depth
        _ = _.view(batch_size, 3, depths_batch_size, height, width)

        src_uv = sampled_tensor.new_empty([batch_size, depths_batch_size, height, width, 2])
        _[:, 2].clamp_(min=div_eps)
        if normalize:
            src_uv[..., 0] = UVModel.normalize_uv(_[:, 0] / _[:, 2], width)
            src_uv[..., 1] = UVModel.normalize_uv(_[:, 1] / _[:, 2], height)
        else:
            src_uv[..., 0] = _[:, 0] / _[:, 2]
            src_uv[..., 1] = _[:, 1] / _[:, 2]
    return src_uv
