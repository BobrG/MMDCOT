import torch


def coords_to_normals(coords):
    """Calculates surface normals using first order finite-differences.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates of the points of shape [**, 3, height, width].

    Returns
    -------
    normals : torch.Tensor
        Surface normals of shape [**, 3, height, width].
    """
    # We will need the input tensor to be a 4D tensor for padding to work. We recover the shape in the end.
    batch_shape = coords.shape[:-3]
    height, width = coords.shape[-2:]
    coords = coords.view(1, -1, 3, height, width)

    dxyz_du = []
    dxyz_dv = []
    for i in range(3):
        d_du = coords[..., i, :, 1:] - coords[..., i, :, :-1]
        d_du = torch.nn.functional.pad(d_du, (0, 1, 0, 0), mode='replicate')
        dxyz_du.append(d_du)

        d_dv = coords[..., i, 1:, :] - coords[..., i, :-1, :]
        d_dv = torch.nn.functional.pad(d_dv, (0, 0, 0, 1), mode='replicate')
        dxyz_dv.append(d_dv)
    dxdu, dydu, dzdu = dxyz_du
    dxdv, dydv, dzdv = dxyz_dv;  del dxyz_du, dxyz_dv, d_du, d_dv

    n_x = dydv * dzdu - dydu * dzdv
    n_y = dzdv * dxdu - dzdu * dxdv
    n_z = dxdv * dydu - dxdu * dydv;  del dxdu, dydu, dzdu, dxdv, dydv, dzdv

    n = torch.stack([n_x, n_y, n_z], dim=-3)
    n = torch.nn.functional.normalize(n, dim=-3)
    return n.view(*batch_shape, 3, height, width)
