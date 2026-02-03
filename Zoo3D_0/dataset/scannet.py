import open3d as o3d
import numpy as np
import os
import cv2
from evaluation.constants import SCANNET200_LABELS, SCANNET200_IDS, SCANNET20_IDS, SCANNET20_LABELS, SCANNET50_IDS, SCANNET50_LABELS


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

class ScanNetDataset:

    def __init__(self, seq_name, data_dir='../data/scannet/posed_images', task='point_cloud', num_classes=200, frame_list=None) -> None:
        self.seq_name = seq_name
        self.segmentation_dir = f'../logs/output/scannet_{task}/{seq_name}/mask'
        self.object_dict_dir = f'../logs/output/scannet_{task}/{seq_name}/object'
        self.scene_data = os.path.join(data_dir, seq_name)
        points_folder = 'points'
        if task == 'posed_images':
            points_folder = 'points_dust3r_posed'
        if task =='unposed_images':
            points_folder = 'points_dust3r_unposed'
        self.mesh_path = os.path.join(data_dir, '..', points_folder, seq_name + '.bin')
        self.num_classes = num_classes
        self.task = task
        self.frame_list = frame_list
        self.depth_scale = 1000.0
        self.image_size = (640, 480)
        self.scannet_image_size = (1280, 960)
    

    def get_frame_list(self):
        if self.frame_list is not None:
            return self.frame_list
        image_list = os.listdir(self.scene_data)
        image_list = [name for name in image_list if name[-4:] == '.jpg']
        image_list = sorted(image_list, key=lambda x: int(x.split('.')[0]))

        frame_id_list = [int(a.split('.')[0]) for a in image_list]
        return list(frame_id_list)
    

    def get_intrinsics(self, frame_id):
        intrinsic_path = f'{self.scene_data}/intrinsic.txt'
        intrinsics = np.loadtxt(intrinsic_path)
        intrinsics = resize_intrinsics(intrinsics, self.scannet_image_size, self.image_size)
        intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinisc_cam_parameters.set_intrinsics(self.image_size[0], self.image_size[1], intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
        return intrinisc_cam_parameters
    

    def get_extrinsic(self, frame_id):
        pose_path = os.path.join(self.scene_data, str(frame_id).zfill(5) + '.txt')
        pose = np.loadtxt(pose_path)
        return pose
    

    def get_depth(self, frame_id):
        depth_path = os.path.join(self.scene_data, str(frame_id).zfill(5) + '.png')
        depth = cv2.imread(depth_path, -1)
        depth = depth / self.depth_scale
        depth = depth.astype(np.float32)
        return depth


    def get_rgb(self, frame_id, change_color=True):
        rgb_path = os.path.join(self.scene_data, str(frame_id).zfill(5) + '.jpg')
        rgb = cv2.imread(rgb_path)

        if change_color:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb    


    def get_segmentation(self, frame_id, align_with_depth=False):
        segmentation_path = os.path.join(self.segmentation_dir, f'{str(frame_id).zfill(5)}.png')
        if not os.path.exists(segmentation_path):
            assert False, f"Segmentation not found: {segmentation_path}"
        segmentation = cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)
        if align_with_depth:
            segmentation = cv2.resize(segmentation, self.image_size, interpolation=cv2.INTER_NEAREST)
        return segmentation


    def get_frame_path(self, frame_id):
        rgb_path = os.path.join(self.rgb_dir, str(frame_id).zfill(5) + '.jpg')
        segmentation_path = os.path.join(self.segmentation_dir, f'{str(frame_id).zfill(5)}.png')
        return rgb_path, segmentation_path
    

    def get_label_features(self):
        label_features_dict = np.load(f'../data/text_features/scannet{self.num_classes}.npy', allow_pickle=True).item()
        return label_features_dict


    def get_scene_points(self):
        vertices = np.fromfile(self.mesh_path, dtype=np.float32).reshape((-1, 6))[:, :3]
        return vertices
    
    
    def get_label_id(self):
        if self.num_classes == 200:
            self.class_id = SCANNET200_IDS
            self.class_label = SCANNET200_LABELS
        elif self.num_classes == 60:
            self.class_id = SCANNET50_IDS
            self.class_label = SCANNET50_LABELS
        elif self.num_classes == 20:
            self.class_id = SCANNET20_IDS
            self.class_label = SCANNET20_LABELS

        self.label2id = {}
        self.id2label = {}
        for label, id in zip(self.class_label, self.class_id):
            self.label2id[label] = id
            self.id2label[id] = label

        return self.label2id, self.id2label
