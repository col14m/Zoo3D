from utils.config import get_dataset, get_args
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import os
import cv2
from concurrent.futures import ThreadPoolExecutor

torch.linalg.inv(torch.eye(5, device="cuda"))

def load_image(path):
    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 480))
    return image

def load_depth(path):
    depth = cv2.imread(str(path), -1) / 1000.
    return depth

def load_pose(path):
    pose = np.loadtxt(path)
    return pose


def unproject_points(x, y, depths, intrinsic, pose):
    mask = depths > 0
    x = x[mask]
    y = y[mask]
    depths = depths[mask]
    if len(x) == 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=depths.device)

    pixel_coords = torch.stack([x, y, torch.ones((len(x)), device=depths.device)])
    normalized_coords = torch.linalg.inv(intrinsic) @ pixel_coords
    camera_points_3d = normalized_coords * depths.unsqueeze(0)
    camera_points_homogeneous = torch.cat([camera_points_3d, torch.ones((1, len(x)), device=depths.device)])
    world_points_homogeneous = pose @ camera_points_homogeneous
    points = (world_points_homogeneous[:3, :] / world_points_homogeneous[3, :]).T

    return points


def filter_occluded_points(points, x, y, depths, intrinsic, pose):
    total_mask = (depths > 0) & (depths < 2.5)
    x = x[total_mask]
    y = y[total_mask]
    depths = depths[total_mask]
    points = points[total_mask]
    unprojected_points = unproject_points(x, y, depths, intrinsic, pose)
    dist = torch.linalg.norm(points - unprojected_points, axis=1)
    mask = dist < 0.02
    total_mask[total_mask.clone()] = mask
    x = x[mask]
    y = y[mask]
    depths = depths[mask]
    points = points[mask]

    return total_mask


def project_points_save(points, intrinsic, pose, depths, hw=(480, 640), filter_occluded=False):
    points = torch.from_numpy(points).to(device='cuda', dtype=torch.float32)
    intrinsic = torch.from_numpy(intrinsic).to(device='cuda', dtype=torch.float32)
    pose = torch.from_numpy(pose).to(device='cuda', dtype=torch.float32)
    depths = torch.from_numpy(depths).to(device='cuda', dtype=torch.float32)
    if points.shape[0] == 0:
        return torch.zeros((0, 2), dtype=torch.float32, device=points.device), \
               torch.zeros((0), dtype=torch.float32, device=points.device)
    points_homogeneous = torch.cat([points, torch.ones((points.shape[0], 1), 
                                    device=points.device)], dim=1).T  # (4, N)
    pose_inv = torch.linalg.inv(pose)
    camera_points_homogeneous = torch.matmul(pose_inv, points_homogeneous)  # (4, N)
    camera_points_3d = camera_points_homogeneous[:3, :] / camera_points_homogeneous[3, :]  # (3, N)
    projected_points = intrinsic @ camera_points_3d  # (3, N)
    projected_points[:2, :] = projected_points[:2, :] / projected_points[2, :]
    x_coords = torch.round(projected_points[0, :]).long()
    y_coords = torch.round(projected_points[1, :]).long()
    height, width = hw
    valid_mask = (x_coords >= 0) & (x_coords < width) & \
                 (y_coords >= 0) & (y_coords < height) & \
                 (projected_points[2, :] > 0)
    x_coords = x_coords 
    y_coords = y_coords 
    points_filtered = points 
    projected_points_filtered = projected_points 
    depth = torch.zeros_like(y_coords).to(dtype=torch.float32, device=depths.device)
    depth[valid_mask] = depths[y_coords[valid_mask], x_coords[valid_mask]]
    if filter_occluded:
        mask = filter_occluded_points(points_filtered, x_coords, y_coords, depth, intrinsic, pose)
    else:
        mask = torch.ones(len(x_coords), dtype=bool, device=depths.device)
    total_mask = valid_mask & mask
    
    return projected_points_filtered[:2, :].cpu(), total_mask.cpu()


def mask_one_idx(x):
    name, data_location, intrinsic, points = x
    name = str(name).zfill(5) 
    pose = load_pose(os.path.join(data_location, f'{name}.txt'))
    depth = load_depth(os.path.join(data_location, f'{name}.png'))
    projected_points, total_mask = project_points_save(points, intrinsic, pose, depth)
    projected_points = projected_points.T
    return projected_points, total_mask

def save_scene_projected_points_and_masks(args):
    dataset = get_dataset(args)
    points = dataset.get_scene_points()
    frame_list = dataset.get_frame_list()
    data_location = dataset.scene_data
    out_dict = []
    camera_intrinsic = dataset.get_intrinsics(0)
    intrinsic = np.asarray(camera_intrinsic.intrinsic_matrix).copy()
    pool_input = [(name, data_location, intrinsic, points) for name in frame_list]
    with ThreadPoolExecutor(max_workers=12) as pool:
        out_dict = list(pool.map(mask_one_idx, pool_input))
    save_points = torch.cat([val[0].unsqueeze(0) for val in out_dict])
    save_masks = torch.cat([val[1].unsqueeze(0) for val in out_dict])
    save_dict = {'points': save_points, 'masks': save_masks}
    out_dir = f'../logs/output/projected_points/scannet_{args.task}'
    os.makedirs(out_dir, exist_ok=True)
    torch.save(save_dict, os.path.join(out_dir, f'{args.seq_name}.pt'))


if __name__ == '__main__':
    args = get_args()
    seq_name_list = args.seq_name_list.split('+')
    for seq_name in tqdm(seq_name_list):
        args.seq_name = seq_name
        save_scene_projected_points_and_masks(args)