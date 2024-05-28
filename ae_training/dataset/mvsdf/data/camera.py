from open3d import camera as _camera
import numpy as np


class PinholeCamera(_camera.PinholeCameraParameters):
    @classmethod
    def from_txt(cls, file, width, height):
        extrinsic = np.loadtxt(file, skiprows=1, max_rows=4)
        intrinsic = np.loadtxt(file, skiprows=7, max_rows=3)

        if (intrinsic[0, 1] != 0) or (intrinsic[1, 0] != 0) or np.any(intrinsic[2] != [0, 0, 1]):
            raise ValueError(f'Only pinhole cameras are supported, got {intrinsic}')
        if np.any(extrinsic[3] != [0, 0, 0, 1]):
            raise ValueError(f'Invalid extrinsic matrix {extrinsic}')

        camera = cls()
        camera.intrinsic.set_intrinsics(
            width=width,
            height=height,
            fx=intrinsic[0, 0],
            cx=intrinsic[0, 2],
            fy=intrinsic[1, 1],
            cy=intrinsic[1, 2])
        camera.extrinsic = extrinsic
        return camera

    @property
    def pos(self):
        R = self.extrinsic[:3, :3]
        t = self.extrinsic[:3, 3]
        return - R.T @ t


def calculate_camera_matrix(intrinsics, extrinsics):
    r"""Calculates camera matrix from intrinsics and extrinsics matrices.

    Parameters
    ----------
    intrinsics : torch.Tensor
        of shape [**, 3, 3]
    extrinsics : torch.Tensor
        of shape [***, 4, 4]. `intrinsics and `extrinsics` must be broadcastable.

    Returns
    -------
    camera_matrix : torch.Tensor
        of shape [****, 4, 4], where **** is the result of broadcasting ** and ***.
    """
    camera_matrix = extrinsics.clone()
    camera_matrix[..., :3, :] = intrinsics @ extrinsics[..., :3, :]
    return camera_matrix
