import torch

class Metric:

    def get_scalar(self, per_pixel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        r'''
        Parameters
        ----------
        per_pixel : torch.Tensor
            of shape [batch_size, height, width]. 
        mask : torch.Tensor
            of shape [batch_size, height, width]. Scalar value is averaged with respect to mask.
        Returns
        -------
        mean_per_pixel_value : torch.Tensor
            Scalar tensor.
        '''
        valids_n = mask.sum([1, 2])
        return (per_pixel.sum([1, 2]) / valids_n).mean()
    
    def visualize_perpixel(self, per_pixel):
        # TODO: Check this function
        pass
        r"""
        Parameters
        ----------
        per_pixel : torch.Tensor
            of shape [batch_size, height, width]. Nans correspond to invalid values.
        Returns
        -------
        per_pixel_visualization : torch.Tensor
            'Hot_r' colomap visualization of per-pixel, of shape [batch_size, 3, height, width],
            with low white corresponding to 0 and black corresponding to self.max_vis_value.
        """
        is_invalid = per_pixel.isnan()

        per_pixel = per_pixel / self.max_vis_value
        vis = self.per_pixel_cmap(per_pixel)[..., :3];  del per_pixel
        # mask invalids with light blue
        vis[is_invalid] = [.5, .5, 1];  del is_invalid

        return torch.from_numpy(vis).permute(0, 3, 1, 2)

    def get_name(self):
        return self.name


class SignedDeviation(Metric):
    def __init__(self):
        self.name = 'Signed_Deviation'
        
    def __call__(self, depth_est: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor, interval: float = 1.0) -> torch.Tensor:
        r'''
        Parameters
        ----------
            depth_est : torch.Tensor
                depth prediction of shape [batch, ch, height, width].
            
            depth_gt : torch.Tensor
                target depth of shape [batch, ch, height, width].
            
            mask : torch.Tensor
                depth mask of shape [batch, ch, height, width].
        Returns
        -------
            error : torch.Tensor 
                batch of per-pixel error computations [batch, height, width].
        '''
        depth_est, depth_gt = depth_est*mask, depth_gt*mask
        error = (depth_est - depth_gt).squeeze(1) / interval

        return error


class IntervalL1Error(Metric):
    def __init__(self):
        self.name = 'L1_Error'
        
    def __call__(self, depth_est: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor, interval: float = 1.0) -> torch.Tensor:
        '''
            Parameters
            ----------
            depth_est : torch.Tensor
                depth prediction of shape [batch, ch, height, width].
            
            depth_gt : torch.Tensor
                target depth of shape [batch, ch, height, width].
            
            mask : torch.Tensor
                depth mask of shape [batch, ch, height, width].
            
            Returns
            -------
                error : torch.Tensor 
                    batch of per-pixel error computations [batch, height, width].
        '''
        depth_est, depth_gt = depth_est*mask, depth_gt*mask
        error = (depth_est - depth_gt).abs() / interval
        
        return error.squeeze(1)


class IntervalL1ErrorThresh(IntervalL1Error):
    def __init__(self, threshold: float):
        super().__init__()
        self.thresh = threshold
        self.name += f'_Thresh_{threshold}'.replace(".", ",")
    
    def __call__(self, depth_est: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor, interval: float = 1.0) -> torch.Tensor:
        return (super().__call__(depth_est, depth_gt, mask, interval) > self.thresh).float()
    
    
