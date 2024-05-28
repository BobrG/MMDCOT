from training.point_tokenizer.point_m2ae import Point_M2AE
from training.dataset.sk3d_dataset import Sk3DDataset

if __name__ == '__main__':
    ds = Sk3DDataset('/mnt/data/g.bobrovskih/sk3d_data/')
    
    print(ds[0])
