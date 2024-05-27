import torch


def smooth_l1_loss(out_depth, gt_depth, mask):
    r"""The loss function for a single stage of CasMVSNet from the original code.
    Although the paper says that the used loss function is L1, in the code it is Huber --- so we use Huber.
    Parameters
    ----------
    out_depth : torch.Tensor
    gt_depth : torch.Tensor
    mask : torch.Tensor
        Zero values indicate unknown depth. The loss is averaged only over nonzero mask values.
    Returns
    -------
    loss : torch.Tensor
    """
    return (torch.nn.functional.smooth_l1_loss(out_depth * mask, gt_depth, reduction='sum')
            / mask.sum())