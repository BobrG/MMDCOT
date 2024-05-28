import torch


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
    crop_start, crop_end = get_center_crop(tensor.shape[-2:], new_size)
    return tensor[..., crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]]


def get_center_crop(old_size, new_size):
    r"""

    Parameters
    ----------
    old_size : iterable of int
        [height, width]
    new_size : iterable of int
        [new_height, new_width]

    Returns
    -------
    crop_start : torch.Tensor
        [top, left]
    crop_end : torch.Tensor
        [bottom, right]
    """
    old_size = torch.as_tensor(old_size)
    new_size = torch.as_tensor(new_size)
    crop = old_size - new_size
    assert (crop >= 0).all()
    crop_start = crop // 2
    crop_end = crop_start + new_size
    return crop_start, crop_end
