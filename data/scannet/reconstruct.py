# based on https://github.com/col14m/TUN3D/blob/main/reconstruction/reconstruct.py

from argparse import ArgumentParser
from pathlib import Path
import open3d as o3d
import numpy as np
import trimesh as tm
import torch
from tqdm import tqdm
import cv2

DUSTER_OUTPUT_SHAPE = (512, 384)
SCANNET_IMG_SHAPE = (1280, 960)
SCANNET_DEPTH_SHAPE = (640, 480)
CONVERT_M_TO_MM = 1000
CONVERT_MM_TO_M = 1 / 1000
UINT8_MAX = 255
VOXEL_SIZE = 0.025
SCANNET_DATA_FOLDER = Path('../data/scannet/posed_images')
INTRINSIC_NAME = 'intrinsic.txt'

def resize_intrinsics(K, base_shape, new_shape):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    fx_new = fx * new_shape[0] / base_shape[0]
    fy_new = fy * new_shape[1] / base_shape[1]
    cx_new = cx * new_shape[0] / base_shape[0]
    cy_new = cy * new_shape[1] / base_shape[1]
    
    return np.array([[fx_new, 0, cx_new, 0], [0, fy_new, cy_new, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def load_dust3r_scene(scene_dir):
    output_params = torch.load(scene_dir / "output_params.pt")
    scene_params = torch.load(scene_dir / "scene_params.pt")

    # load glb
    scene_mesh = tm.load(str(scene_dir / "scene.glb"))
    return output_params, scene_params, scene_mesh


def get_pcd_min_max(path):
    try:
        data = np.loadtxt(path)
    except Exception as e:
        print("Can't load txt file, trying to read line by line")
        data = []
        with open(path, "r") as fin:
            lines = fin.readlines()
            for i, line in enumerate(lines):
                try:
                    data.append(list(map(float, line.split(" "))))
                except Exception as e:
                    print(f"Error at {i}: {line} ({e}), skipping")
        data = np.array(data)
    vertices = data[:, :3]
    return vertices.min(axis=0), vertices.max(axis=0)


def process_scene_scannet(data):
    scene_id, recs_path, out_path, confidence_trunc = data
    out_path = Path(out_path) / scene_id
    if Path(out_path / f"{scene_id}.ply").exists():
        return

    output_params, scene_params, scene_mesh = load_dust3r_scene(recs_path / scene_id)

    poses = scene_params["poses"]
    pts3d = scene_params["pts3d"]
    Ks = scene_params["Ks"]
    depths = scene_params['depths']
    image_names = scene_params["image_files"]
    confs = scene_params["im_conf"]

    np.savetxt(out_path / INTRINSIC_NAME, resize_intrinsics(Ks.mean(0), DUSTER_OUTPUT_SHAPE, SCANNET_IMG_SHAPE))

    vertices = np.empty((0, 3))
    colors = np.empty((0, 3))

    for i, image_name in enumerate(image_names):
        
        name = image_name

        def get_image_name(image_name):
            return str(int(image_name.stem)).zfill(5)
        image_name = Path(image_name)
        # save image
        shutil.copy(SCANNET_DATA_FOLDER / scene_id / image_name.name, out_path / (get_image_name(image_name) + '.jpg'))
        # save depth
        depth_map = depths[i].numpy() * CONVERT_M_TO_MM
        depth_map = cv2.resize(depth_map, SCANNET_DEPTH_SHAPE, interpolation=cv2.INTER_NEAREST)
        depth_map = depth_map.astype(np.uint16)
        cv2.imwrite(str(out_path / (get_image_name(image_name) + '.png')), depth_map)
        # save pose
        pose = poses[i].numpy()
        np.savetxt(str(out_path / (get_image_name(image_name) + '.txt')), pose)
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ch, cw = confs[i].shape
        image = cv2.resize(image, (cw, ch))
        if confidence_trunc > 0:
            conf = confs[i].numpy().astype(np.float32)
            conf = conf > confidence_trunc
        else:
            conf = np.ones_like(confs[i], dtype=bool)

        pts3d_i = pts3d[i][conf].numpy()
        colors_i = image[conf, :].astype(np.float32)
        vertices = np.concatenate([vertices, pts3d_i], axis=0)
        colors = np.concatenate([colors, colors_i], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(colors / UINT8_MAX)
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

    o3d.io.write_point_cloud(str(out_path / f"{scene_id}.ply"), pcd)


def process_scene_align_scannet(data):

    scene_id, recs_path, out_path, depth_trunc, confidence_trunc = data
    out_path = Path(out_path) / scene_id
    if Path(out_path / f"{scene_id}.ply").exists():
        return
    output_params, scene_params, scene_mesh = load_dust3r_scene(recs_path / scene_id)
    poses = scene_params["poses"]
    depths = scene_params["depths"]
    Ks = scene_params["Ks"]


    np.savetxt(out_path / INTRINSIC_NAME, resize_intrinsics(Ks.mean(0), DUSTER_OUTPUT_SHAPE, SCANNET_IMG_SHAPE))
    
    image_names = scene_params["image_files"]

    image_file = Path(image_names[0])

    gt_depth = image_file.parent / (image_file.stem + ".png")
    gt_pose = image_file.parent / (image_file.stem + ".txt")

    gt_depth = cv2.imread(str(gt_depth), -1) * CONVERT_MM_TO_M
    gt_pose = np.loadtxt(str(gt_pose))

    h, w = gt_depth.shape
    depth = depths[0].numpy().astype(np.float32)
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    valid_mask = (gt_depth > 1e-10) & (depth > 1e-10)
    depth = depth[valid_mask]
    gt_depth = gt_depth[valid_mask]

    scale = np.median(gt_depth / depth)

    first_pose = np.linalg.inv(poses[0].numpy())
    tsdffusion = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_SIZE,
        sdf_trunc=0.1,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i, image_name in enumerate(image_names):
        name = image_name
        def get_image_name(image_name):
            return str(int(image_name.stem)).zfill(5)
        image_name = Path(image_name)
        # save image
        shutil.copy(SCANNET_DATA_FOLDER / scene_id / image_name.name, out_path / (get_image_name(image_name) + '.jpg'))
        # save scaled depth
        depth_map = depths[i].numpy() * CONVERT_M_TO_MM * scale
        depth_map = cv2.resize(depth_map, SCANNET_DEPTH_SHAPE, interpolation=cv2.INTER_NEAREST)
        depth_map = depth_map.astype(np.uint16)
        cv2.imwrite(str(out_path / (get_image_name(image_name) + '.png')), depth_map)


        color = o3d.io.read_image(name)

        fx, fy = Ks[i][0, 0], Ks[i][1, 1]
        cx, cy = Ks[i][0, 2], Ks[i][1, 2]
        depth = depths[i].numpy().astype(np.float32) * scale
        if confidence_trunc > 0:
            confidence = scene_params["im_conf"][i]
            confidence = confidence.numpy().astype(np.float32)
            depth[confidence < confidence_trunc] = 0

        h, w = np.asarray(color).shape[:2]
        dh, dw = depth.shape

        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth_o3d,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )
        camera_o3d = o3d.camera.PinholeCameraIntrinsic(
            w, h, fx / dw * w, fy / dh * h, cx / dw * w, cy / dh * h
        )
        # save pose
        pose = poses[i].numpy()
        pose = first_pose @ pose
        pose[:3, 3] *= scale
        pose = gt_pose @ pose
        np.savetxt(str(out_path / (get_image_name(image_name) + '.txt')), pose)

        tsdffusion.integrate(
            rgbd,
            camera_o3d,
            np.linalg.inv(pose),
        )
    pc = tsdffusion.extract_point_cloud()
    pc.voxel_down_sample(voxel_size=VOXEL_SIZE)
    o3d.io.write_point_cloud(str(out_path / f"{scene_id}.ply"), pc)


def check_valid_input(path):
    if not (path / "output_params.pt").exists():
        return False
    if not (path / "scene_params.pt").exists():
        return False
    if not (path / "scene.glb").exists():
        return False
    return True


def process_scannet(parser):
    args = parser.parse_args()

    out_path = Path(args.out_path)
    if args.align:
        out_path = out_path / "rec_unposed_images"
    else:
        out_path = out_path / "rec_posed_images"
    out_path.mkdir(parents=True, exist_ok=True)
    recs_path = Path(args.recs_path)
    scene_ids = sorted([*recs_path.glob("*")])
    scene_ids = list(map(lambda x: Path(x).stem, scene_ids))

    if args.align:
        data = [
            (scene_id, recs_path, out_path, args.depth_trunc, args.confidence_trunc)
            for scene_id in scene_ids[args.job_id : args.job_id_upper : args.num_jobs]
            if check_valid_input(recs_path / scene_id)
        ]
        [*tqdm(map(process_scene_align_scannet, data), total=len(data))]
    else:
        data = [
            (scene_id, recs_path, out_path, args.confidence_trunc)
            for scene_id in scene_ids[args.job_id : args.job_id_upper : args.num_jobs]
            if check_valid_input(recs_path / scene_id)
        ]
        [*tqdm(map(process_scene_scannet, data), total=len(data))]


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-rp", "--recs_path", type=str, required=True)
    parser.add_argument(
        "-o", "--out_path", type=str, default="../data/scannet"
    )
    parser.add_argument("-a", "--align", action="store_true")
    parser.add_argument("-dt", "--depth_trunc", type=float, default=3.0)
    parser.add_argument("-ct", "--confidence_trunc", type=float, default=0.0)
    parser.add_argument("-i", "--job_id", type=int, default=0)
    parser.add_argument("-u", "--job_id_upper", type=int, default=int(1e10))
    parser.add_argument("-n", "--num_jobs", type=int, default=1)
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, choices=["scannet"]
    )

    args = parser.parse_args()

    if args.dataset == "scannet":
        process_scannet(parser)