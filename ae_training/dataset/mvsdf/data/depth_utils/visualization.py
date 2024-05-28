import torch

from .reprojections import depth_to_absolute_coordinates
from .normals import coords_to_normals

# TODO make more pytorch-y, i.e no implicit conversions etc


def render_depth_map(depth, depth_type, K=None, calibration=None, light_pos=None, light_dir=None):
    """Render the surface corresponding to the depth map using simple occlusion-less Lambertian rendering.

    Parameters
    ----------
    depth : array_like
        Depth map of shape (h, w) or tensor of depth maps of shape (**, 1, h, w).
    depth_type : str
        Type of the depth map, one of 'perspective' -- meaning the distance from point to camera,
         'orthogonal' -- meaning the distance from point to image plane, or 'disparity'.
    K : array_like, optional
        Camera projection matrix.
    calibration : dict, optional if `type` is not 'disparity'
        Intrinsic parameters of the camera:
            cx, cy -- coordinates of the principal point in the image coordinate system in pixels,
            f -- focal length in pixels,
            baseline, required for disparity -- baseline of the camera in metric units.
        Either `K` or `calibration` is required.
    light_pos : array_like, optional
        Position of the light source in the camera coordinate system, broadcastable to (**, 3),
         where X is pointing rightward, Y is pointing downward, Z is pointing forward.
        Mutually exclusive with `to_light`.
    light_dir : array_like, optional
        Direction of the lighting in the camera coordinate system, broadcastable to (**, 3).
        Mutually exclusive with `light_pos`.

    Returns
    -------
    rendering : torch.Tensor
        Rendered image of the surface in the form of (**, 1, h, w) tensor of shading values in the range [-1,1].
    """
    assert (light_pos is not None) != (light_dir is not None), 'Use either `light_pos` or `to_light`, not both.'

    depth = torch.as_tensor(depth)
    dtype = depth.dtype
    if K is not None:
        K = torch.as_tensor(K, dtype=dtype)
    else:
        K = torch.zeros(3, 3, dtype=dtype)
        K[0, 0] = K[1, 1] = float(calibration['f'])
        K[2, 2] = 1
        K[0, 2] = float(calibration['cx'])
        K[1, 2] = float(calibration['cy'])
    if depth_type == 'disparity':
        baseline = calibration['baseline']
    else:
        baseline = None
    surface_coords = depth_to_absolute_coordinates(depth, depth_type=depth_type,
                                                   intrinsic_matrix=K, stereo_baseline=baseline)
    surface_normals = coords_to_normals(surface_coords)

    if light_dir is not None:
        to_light = -torch.as_tensor(light_dir).to(surface_normals)
        to_light = torch.nn.functional.normalize(to_light, dim=-1)
        surface_normals, to_light = torch.broadcast_tensors(surface_normals, to_light.unsqueeze(-1).unsqueeze(-1))
        shading = torch.sum(surface_normals * to_light, dim=-3, keepdim=True)
    else:
        light_pos = torch.as_tensor(light_pos).to(surface_coords)
        surface_coords, light_pos = torch.broadcast_tensors(surface_coords, light_pos.unsqueeze(-1).unsqueeze(-1))
        to_light = torch.nn.functional.normalize(light_pos - surface_coords, dim=-3)
        shading = torch.sum(surface_normals * to_light, dim=-3, keepdim=True)

    return shading
