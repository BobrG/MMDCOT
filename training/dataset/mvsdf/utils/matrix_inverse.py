import torch


def kludgy_matrix_inverse(matrix):
    r"""Implements batch matrix inversion, i.e torch.inverse.

    If the total batch size is not 2, calls the actual torch.inverse.
    Otherwise, does inversion one by one and stacks the results.
    See this discussion for explanation
    https://discuss.pytorch.org/t/data-corruption-with-matrix-inverse-after-other-tensor-to-cuda-non-blocking-true/104065

    Parameters
    ----------
    matrix : torch.Tensor
        of shape [**, n, n]

    Returns
    -------
    inverted_matrix : torch.Tensor
        of shape [**, n, n]
    """
    batch_shape = matrix.shape[:-2]
    n = matrix.shape[-1]
    matrix = matrix.view(-1, n, n)
    batch_size = matrix.shape[0]
    if batch_size != 2:
        matrix = matrix.view(*batch_shape, n, n)
        return matrix.inverse()
    else:
        inverted_matrix = torch.stack([_.inverse() for _ in matrix])
        return inverted_matrix.view(*batch_shape, n, n)
