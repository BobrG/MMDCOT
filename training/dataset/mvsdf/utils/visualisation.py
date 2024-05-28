import torch
import numpy as np
from matplotlib import cm

VMIN, VMAX = -3, 3 # in intervals, synced with worst Thresh metric

def preprocess_image(name: str, img: torch.Tensor):
    '''
    Parameters
    ----------
    img : torch.Tensor
        of shape [height, width] or [channels, height, width]
    name : str
        img / depth / normalmap / errormap
    Returns
    -------
    img : torch.Tensor
        of shape [3, height, width].
     
    '''
        
    img = img.detach().cpu()
    if 'errormap' in name:
        img = (img - VMIN) / (VMAX - VMIN)
        if 'signed' not in name:
            img = cm.jet(img.numpy())[..., :3]
        else:
            img = cm.bwr(img.numpy())[..., :3]
        img = torch.from_numpy(img) 
        img = img.mul_(255).round_().clamp_(0, 255).permute(2, 0, 1)
    elif 'depth' in name:
        img = cm.plasma(img.numpy())[..., :3]
        img = torch.from_numpy(img) 
        img = img.mul_(255).round_().clamp_(0, 255).permute(2, 0, 1)
    elif 'img' in name or 'normalmap' in name:
        img = img.mul_(255).round_().clamp_(0, 255)
    elif 'tsdf' in name:
        new_img = []
        views, depths, h, w = img.shape
        for view in range(views):
            for d in range(depths):
                vmin, vmax = -0.1, 0.1
                a = (img[view, d].numpy() - vmin) / (vmax - vmin)
                a = torch.from_numpy(cm.bwr(a)[..., :3]).mul_(255).round_().clamp_(0, 255).permute(2, 0, 1).unsqueeze(0)
                new_img.append(a)
        img = torch.cat(new_img, dim=0)

    elif 'conf' in name:
        img = cm.hot(img.numpy())[..., :3]
        img = torch.from_numpy(img) 
        img = img.mul_(255).round_().clamp_(0, 255).permute(2, 0, 1)
    
    return img

def draw_img(img):
    if isinstance(img, np.ndarray):
        return preprocess_image('img', torch.from_numpy(img))
    elif isinstance(img, torch.Tensor):
        return preprocess_image('img', img)
    
def draw_depth(img):
    if isinstance(img, np.ndarray):
        return preprocess_image('depth', torch.from_numpy(img)).permute(1, 2, 0)
    elif isinstance(img, torch.Tensor):
        return preprocess_image('depth', img).permute(1, 2, 0)
    
def get_pseudo_normals(depthmap, scale=10):
    r"""
    Parameters
    ----------
    depthmap : torch.Tensor
        of shape [batch_size, 1, height, width]
    scale : float
    Returns
    -------
    normals : torch.Tensor
        of shape [batch_size, 3, height, width], with coordinates in range [0, 1].
    """
    shape = list(depthmap.shape)
    shape[1] = 3
    normals = depthmap.new_empty(shape)

    depthmap = torch.nn.functional.pad(depthmap, (1, 1, 1, 1), 'replicate')
    normals[..., 0:1, :, :] = depthmap[..., 1:-1, 2:] - depthmap[..., 1:-1, 1:-1]
    normals[..., 1:2, :, :] = depthmap[..., 1:-1, 1:-1] - depthmap[..., 2:, 1:-1]
    normals[..., 2:, :, :] = 1 / scale
    normals = torch.nn.functional.normalize(normals, dim=-3)
    normals[..., :2, :, :] = (normals[..., :2, :, :] + 1).div_(2)
    return normals

