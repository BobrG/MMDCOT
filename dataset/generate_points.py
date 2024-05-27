import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from torchvision.transforms import functional as TF
from mvsdf.data.depth_utils.reprojections import depth_to_absolute_coordinates, ref_depth_to_src_uv

def unpack_float32(ar):
    shape = ar.shape[:-1]
    return ar.ravel().view(np.float32).reshape(shape)

def load_cam(fname):
    with open(fname) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    extrinsics = torch.FloatTensor(list(map(float, ' '.join(lines[1:5]).split(' ')))).view(4, 4)
    intrinsics = torch.FloatTensor(list(map(float, ' '.join(lines[7:10]).split(' ')))).view(3, 3)
    return extrinsics, intrinsics

def crop_and_resize(depth, intrinsics, crop_coords, new_hw):
    h_min, h_max, w_min, w_max = crop_coords
    depth_cropped = depth[..., h_min:h_max, w_min:w_max]
    depth_resized = TF.resize(depth_cropped.unsqueeze(0), new_hw, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
    intrinsics_adjusted = intrinsics.clone()
    intrinsics_adjusted[:, -1] -= torch.tensor([w_min, h_min, 0], dtype=intrinsics.dtype)
    orig_h, orig_w = h_max - h_min, w_max - w_min
    scale_h, scale_w = new_hw[0] / orig_h, new_hw[1] / orig_w
    scaling_matrix = torch.diag(torch.tensor([scale_w, scale_h, 1], dtype=intrinsics.dtype))
    intrinsics_adjusted = scaling_matrix @ intrinsics_adjusted
    return depth_resized, intrinsics_adjusted

def filter_non_zero_points(pc, mask):
    return pc[mask]

def random_sample(pc, num=2048):
    npoints = pc.shape[0]
    permutation = np.arange(npoints)
    np.random.shuffle(permutation)
    pc = pc[permutation[:num]]
    return pc

def pc_norm(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def process_point_cloud(pc, mask, num_samples=2048):
    filtered_pc = filter_non_zero_points(pc, mask)
    sampled_pc = random_sample(filtered_pc, num_samples)
    normalized_pc = pc_norm(sampled_pc)
    return normalized_pc

def process_object(dataset_path, object_name):
    depth_filename = os.path.join(dataset_path, 'addons', object_name, 'proj_depth', 'stl.clean_rec.aa@tis_right.undist', '0000.png')
    cam_filename = os.path.join(dataset_path, 'addons', object_name, 'tis_right', 'rgb', 'mvsnet_input', '00000000_cam.txt')
    
    if not os.path.exists(depth_filename) or not os.path.exists(cam_filename):
        print(f"Files for {object_name} not found, skipping.")
        return

    depth = torch.FloatTensor(unpack_float32(np.asarray(Image.open(depth_filename))).copy())
    mask = (depth >= 0).to(torch.float32)
    depth = depth.where(mask != 0, depth.new_tensor(0))

    extrinsics, intrinsics = load_cam(cam_filename)
    crop_coords = [601, 1952, 291, 1887]
    new_hw = [576, 768]
    depth, intrinsics = crop_and_resize(depth, intrinsics, crop_coords, new_hw)
    mask = (depth != 0)

    pc = depth_to_absolute_coordinates(depth.unsqueeze(0), 'orthogonal', intrinsics.unsqueeze(0)).reshape(3, -1)
    pc = (extrinsics.inverse() @ torch.cat((pc, torch.ones(1, pc.shape[1])), dim=0))[:-1].detach().cpu().numpy().T

    pc = process_point_cloud(pc, mask.reshape(-1), num_samples=2048)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    ply_filename = os.path.join(dataset_path, 'point_clouds', f'{object_name}_pc.ply')
    o3d.io.write_point_cloud(ply_filename, pcd)
    print(f"Processed {object_name}, saved to {ply_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process objects to generate point clouds.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')

    args = parser.parse_args()
    dataset_path = args.dataset_path

    os.makedirs(os.path.join(dataset_path, 'point_clouds'), exist_ok=True)
    objects_dir = os.path.join(dataset_path, 'addons')
    object_names = [name for name in os.listdir(objects_dir) if os.path.isdir(os.path.join(objects_dir, name))]

    for object_name in object_names:
        process_object(dataset_path, object_name)
