# MMDCOT

For all experiment please clone this repo: ```git clone https://github.com/BobrG/MMDCOT```

### Early experiments

* 3D tokenizer:
  Before applying tokenizer to 3D point cloud data, it was debugged following guidelines from the official repo of the picked model - [Point Masked AutoEncoder](https://github.com/ZrrSkywalker/Point-M2AE/tree/main). The quality of tokenizer was checked by visualization of produced reconstructions on ShapeNet dataset. To reproduce this experiment please follow these steps:
  1. prepare Point_M2AE code and download ShapeNet data as described in the original repo
  
  2. replace /path/to/Point-M2AE/models/Point_M2AE.py with /path/to/MMDCOT/Point_M2AE.py and create directory /path/to/Point-M2AE/pointmae_reconstructions
 
  3. launch pre-training on ShapeNet with the script:
      ```
      CUDA_VISIBLE_DEVICES=0 python main.py --config /path/to/MMDCOT/point-m2ae.yaml --exp_name pre-train --start_ckpts ckpts/pre-train.pth --ckpt ckpts/pre-train.pth
      ```
  4. Now you are able to visualize reconstructions and original data, stored in /path/to/Point-M2AE/pointmae_reconstructions as plys via i.e meshlab.
  
