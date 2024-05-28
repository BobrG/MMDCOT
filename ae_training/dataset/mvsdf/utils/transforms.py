import torch
from PIL import Image
from typing import Tuple, List

from torchvision import transforms
import torchvision.transforms.functional as TF

from mvsdf.data.image_coordinates_model.default import UVModel


class MultiCompose(transforms.Compose):
    '''
    An extension to torchvision.Compose class. Operates with arbitrary number of inputs and applies same transforms to them.
    Initialized with number of tranforms that take arbitrary number of inputs, instead of single image.
    
    '''

    def __call__(self, **kwargs):
        for t in self.transforms:
            kwargs = t(**kwargs)
        return kwargs.values()

    
class ToTensor(transforms.ToTensor):
    def __call__(self, image, intrinsics, depth, mask) -> torch.Tensor:
        return tuple(TF.to_tensor(var) for name, var in vars().items() if (name != 'self' and var is not None))


class CropTransform():
    '''
        Crop image, intrinsics and depth at one place.
    '''
        
    def __init__(self, crop: List) -> None:
        self.h_min, self.h_max, self.w_min, self.w_max = crop
        
    def __call__(self, **kwargs) -> Tuple[torch.Tensor, ...]:
        res = dict()
        for data_name, _ in kwargs.items():
            if kwargs[data_name] is None:
                res[data_name] = None
                continue
            
            if any([name in data_name for name in ['depth', 'mask', 'image']]):
                res[data_name] = kwargs[data_name][..., self.h_min:self.h_max, self.w_min:self.w_max]
            elif 'intr' in data_name:
                new_intr = kwargs[data_name].clone()
                new_intr[:, -1] = new_intr[..., -1] - torch.FloatTensor([self.w_min, self.h_min, 0])
                res[data_name] = new_intr; del new_intr
        
        assert len(res.keys()) > 0, 'transform cannot be applied to nothing'
        return res


class ResizeTransform():
    '''
        Resize image, intrinsics and depth at one place.
    '''
    
    def __init__(self, new_hw: List) -> None:
        self.height, self.width = new_hw

    # transforms for image and intrinsics
    def __call__(self, **kwargs) -> Tuple[torch.Tensor, ...]:
        res = dict()
        h_orig, w_orig = None, None
        for data_name, _ in kwargs.items():
            if kwargs[data_name] is None:
                res[data_name] = None
                continue
            
            if any([name in data_name for name in ['depth', 'mask']]):
                res[data_name] = UVModel.interpolate(kwargs[data_name].unsqueeze(0).unsqueeze(0), (self.height, self.width), mode='nearest').squeeze(0).squeeze(0)
            elif 'image' in data_name:
                h_orig, w_orig = kwargs[data_name].shape[1:]
                res[data_name] = UVModel.interpolate(kwargs[data_name].unsqueeze(0), (self.height, self.width), mode='bilinear').squeeze(0)
            elif 'intrinsics' in data_name:
                if h_orig is None or w_orig is None:
                    raise NotImplementedError
                res['intrinsics'] = torch.diag(torch.FloatTensor([self.width / w_orig, self.height / h_orig, 1])) @ kwargs['intrinsics']
              
        assert len(res.keys()) > 0, 'transform cannot be applied to nothing'  
        return res
        
        