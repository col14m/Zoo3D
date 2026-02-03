import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import cv2
import open_clip
import argparse
from concurrent.futures import ThreadPoolExecutor
from mmdet3d.structures import rotation_3d_in_axis
import threading
import time
from utils.config import get_dataset, get_args

########
import sys
import warnings
warnings.simplefilter('ignore')
sys.path.append('third_party/sam2')
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import torch.nn.functional as F
from evaluation.constants import SCANNET200_LABELS, SCANNET20_LABELS, SCANNET50_LABELS

H, W = 480, 640
MASK_BS = 10
SAM_DELTA = 5
CLIP_EMB_DIM = 1024

def compute_label_id_by_sim(inst_features, label_text_features, label_names, label_ids, valid_class_indices=None, top_k=5, norm=False):

    inst_num_bf = inst_features.shape[0]

    # Step 1: for each given instance, get the most possible label index by computing similarity
    if norm:
        inst_features = inst_features / np.linalg.norm(inst_features, axis=-1, keepdims=True)
        label_text_features = label_text_features / np.linalg.norm(label_text_features, axis=-1, keepdims=True)

    raw_similarity = np.dot(inst_features, label_text_features.T)
    exp_sim = np.exp(raw_similarity * 100)
    prob = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
    label_indices = np.argmax(prob, axis=-1)

    if valid_class_indices is not None:
        inst_kept_mask = np.isin(label_indices, valid_class_indices)
        label_indices = np.where(inst_kept_mask, label_indices, -1 * np.ones_like(label_indices))
        kept_inst_indices = np.where(inst_kept_mask)[0]
    else:
        kept_inst_indices = np.arange(inst_num_bf)

    # Step 2: get the most possible label_name and label_ID for each given instance
    label_indices_list = label_indices.tolist()

    kept_inst_indices = kept_inst_indices.tolist()

    inst_label_names_list = []
    inst_label_ids_list = []
    for i, label_index in enumerate(label_indices_list):
        if label_index != -1:
            inst_label_names_list.append( label_names[label_index] )
            inst_label_ids_list.append( label_ids[label_index] )
        else:
            inst_label_names_list.append("unlabeled")
            inst_label_ids_list.append(-1)

    return inst_label_ids_list, inst_label_names_list, prob.max(-1)


def load_clip():
    print(f'[INFO] loading CLIP model...')
    # model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    open_clip_model = 'ViT-H-14-quickgelu'
    open_clip_pretrained = 'dfn5b'
    model, _, preprocess = open_clip.create_model_and_transforms(open_clip_model, pretrained=open_clip_pretrained)

    model.cuda()
    model.eval()
    print(f'[INFO]', ' finish loading CLIP model...')
    return model, preprocess

CROP_SCALES = 3

def get_cropped_image(mask, rgb):
    '''
        Given a mask and an rgb image, we crop the image with CROP_SCALES scales based on the mask.
    '''
    def mask2box_multi_level(mask, level, expansion_ratio):
        pos = np.where(mask)
        top = np.min(pos[0])
        bottom = np.max(pos[0])
        left = np.min(pos[1])
        right = np.max(pos[1])

        if level == 0:
            return left, top, right , bottom
        shape = mask.shape
        x_exp = int(abs(right - left)*expansion_ratio) * level
        y_exp = int(abs(bottom - top)*expansion_ratio) * level
        return max(0, left - x_exp - 1), max(0, top - y_exp - 1), min(shape[1], right + x_exp + 1), min(shape[0], bottom + y_exp + 1)

    def crop_image(rgb, mask):
        multiscale_cropped_images = []
        for level in range(CROP_SCALES):
            left, top, right, bottom = mask2box_multi_level(mask, level, 0.1)
            cropped_image = rgb[top:bottom, left:right].copy()
            if cropped_image.sum() != 0:
                multiscale_cropped_images.append(cropped_image)
        return multiscale_cropped_images

    mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    multiscale_cropped_images = crop_image(rgb, mask)
    return multiscale_cropped_images
        
def pad_into_square(image):
    width, height = image.size
    new_size = max(width, height)
    new_image = Image.new("RGB", (new_size, new_size), (255,255,255))
    left = (new_size - width) // 2
    top = (new_size - height) // 2
    new_image.paste(image, (left, top))
    return new_image


def get_face_distances(points, boxes):
    """Calculate distances from point to box faces.

    Args:
        points (Tensor): Final locations of shape (N_points, N_boxes, 3).
        boxes (Tensor): 3D boxes of shape (N_points, N_boxes, 7)

    Returns:
        Tensor: Face distances of shape (N_points, N_boxes, 6),
        (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).
    """
    # If boxes have 6 dimensions, add a zero in the 7th column
    if boxes.shape[-1] == 6:
        boxes = torch.cat([boxes, torch.zeros(*boxes.shape[:-1], 1, dtype=boxes.dtype, device=boxes.device)], dim=-1)
    shift = torch.stack(
        (points[..., 0] - boxes[..., 0], points[..., 1] - boxes[..., 1],
            points[..., 2] - boxes[..., 2]),
        dim=-1).permute(1, 0, 2)
    shift = rotation_3d_in_axis(
        shift, -boxes[0, :, 6], axis=2).permute(1, 0, 2)
    centers = boxes[..., :3] + shift
    dx_min = centers[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
    dx_max = boxes[..., 0] + boxes[..., 3] / 2 - centers[..., 0]
    dy_min = centers[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
    dy_max = boxes[..., 1] + boxes[..., 4] / 2 - centers[..., 1]
    dz_min = centers[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
    dz_max = boxes[..., 2] + boxes[..., 5] / 2 - centers[..., 2]
    return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max),
                        dim=-1)


def get_masks(points, boxes):
    src_points = points.to(boxes.device)
    n_points = src_points.shape[0]
    n_boxes = boxes.shape[0]
    new_point = src_points.unsqueeze(1).expand(n_points, n_boxes, 3)
    new_bboxes = boxes.unsqueeze(0).expand(n_points, n_boxes, 
                                        boxes.shape[1])
    
    face_distances = get_face_distances(new_point, new_bboxes)
    seg_mask = face_distances.min(dim=-1).values > 0
    return seg_mask.T


def get_bboxes_fully_vectorized(masks):
    F, N, H, W = masks.shape
    

    active_rows = torch.any(masks, dim=3)  # (F, N, H)
    active_cols = torch.any(masks, dim=2)  # (F, N, W)
    

    y_min = torch.argmax(active_rows.float(), dim=2)
    x_min = torch.argmax(active_cols.float(), dim=2)
    

    y_max = H - 1 - torch.argmax(active_rows.flip(dims=[2]).float(), dim=2)
    x_max = W - 1 - torch.argmax(active_cols.flip(dims=[2]).float(), dim=2)
    

    empty_masks = ~torch.any(torch.any(masks, dim=3), dim=2)
    

    y_min = torch.where(empty_masks, torch.tensor(0, device=masks.device), y_min)
    x_min = torch.where(empty_masks, torch.tensor(0, device=masks.device), x_min)
    y_max = torch.where(empty_masks, torch.tensor(0, device=masks.device), y_max)
    x_max = torch.where(empty_masks, torch.tensor(0, device=masks.device), x_max)
    
    bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    
    return bboxes
    
LABELS = SCANNET200_LABELS

class FastCLIPFeatures:
    def __init__(self, args, dense_mask=True):
        text_label_features = np.load(f'../data/text_features/scannet{args.num_classes}.npy', allow_pickle=True).item()
        text_label_features = [text_label_features[label] for label in LABELS]
        self.text_label_features = np.stack(text_label_features, axis=0)
        self.model, self.preprocess = load_clip()
        if dense_mask:
            sam2_checkpoint = "third_party/sam2/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
            self.sam_model = SAM2ImagePredictor(sam2_image_model)
        else:
            self.sam_model = None

    def __call__(self, dataset, masks):
        inst_features = []
        return_mask = []
        for i in range(0, len(masks), MASK_BS):
            inst_features_p, return_mask_p = self.get_mask_crops(dataset, masks[i: i + MASK_BS, :], self.preprocess)
            inst_features.extend(inst_features_p)
            return_mask.extend(return_mask_p)
        inst_features = np.array(inst_features)
        return_mask = np.array(return_mask)
        dataset.get_label_id()
        inst_ids, inst_names, probs = compute_label_id_by_sim(inst_features, self.text_label_features, dataset.class_label, dataset.class_id)
        return torch.tensor(inst_ids), torch.from_numpy(probs), torch.from_numpy(return_mask)

    def get_mask_crops(self, dataset, masks, preprocess, top_k=5, sort_key='bbox_size'):
        scene_id = dataset.seq_name
        images = [np.array(cv2.resize(dataset.get_rgb(frame_id), (W, H))) for frame_id in dataset.get_frame_list()]
        dict_path = f'../logs/output/projected_points/scannet_{dataset.task}/{scene_id}.pt'
        dict_ = torch.load(dict_path, weights_only=False)
        points, good_masks = dict_['points'].to('cuda'), dict_['masks'].to('cuda')
        all_masks = (good_masks.unsqueeze(0) & masks.unsqueeze(1).to(torch.bool)).unsqueeze(-1)
        points = points.unsqueeze(0)
        masked_points = torch.where(all_masks, points, float('nan'))
        x_coords = masked_points[:, :, :, 0]
        y_coords = masked_points[:, :, :, 1]
        
        x_min = torch.nanquantile(x_coords, 0, dim=-1)  # (num_masks, num_frames)
        y_min = torch.nanquantile(y_coords, 0, dim=-1)  # (num_masks, num_frames)
        x_max = torch.nanquantile(x_coords, 1, dim=-1)  # (num_masks, num_frames)
        y_max = torch.nanquantile(y_coords, 1, dim=-1)  # (num_masks, num_frames)
        x_min = torch.nan_to_num(x_min, nan=0.0)
        y_min = torch.nan_to_num(y_min, nan=0.0)
        x_max = torch.nan_to_num(x_max, nan=-SAM_DELTA)
        y_max = torch.nan_to_num(y_max, nan=-SAM_DELTA)
        x_min = (x_min - SAM_DELTA).clamp(0, W)
        y_min = (y_min - SAM_DELTA).clamp(0, H)
        x_max = (x_max + SAM_DELTA).clamp(0, W)
        y_max = (y_max + SAM_DELTA).clamp(0, H)
        bbox_coords = torch.stack([x_min, y_min, x_max, y_max], dim=-1)  # num_masks x num_frames x 4
        masked_points_2 = torch.where(all_masks, points, 1e10)
        x_coords = torch.round(masked_points_2[:, :, :, 0]).long()  # num_masks x num_frames x num_points
        y_coords = torch.round(masked_points_2[:, :, :, 1]).long()  # num_masks x num_frames x num_points
        valid_mask = (x_coords != 1e10) & (y_coords != 1e10)


        batch_idx, mask_idx, point_idx = torch.where(valid_mask)


        x_valid = x_coords[batch_idx, mask_idx, point_idx]
        y_valid = y_coords[batch_idx, mask_idx, point_idx]
        mask_2d = torch.zeros((len(masks), len(images), H, W), dtype=torch.bool, device=points.device)  # num_masks x num_frames x H x W
        mask_2d[batch_idx, mask_idx, y_valid, x_valid] = True

        if self.sam_model is not None:
            bbox_prompt = [bbox_coord.cpu().numpy() for bbox_coord in bbox_coords.transpose(0, 1)]
            self.sam_model.set_image_batch(images)
            masks, _, _ = self.sam_model.predict_batch(box_batch=bbox_prompt, multimask_output=False)
            mask_2d = torch.from_numpy(np.stack(masks).reshape((len(images), -1, H, W)).astype(np.bool_)).to(device=bbox_coords.device).transpose(0, 1) # num_frames x num_masks x H x W
            mask_2d[bbox_coords.sum(-1) == 0, :, :] = False
            bbox_coords = get_bboxes_fully_vectorized(mask_2d)
        volumes = (bbox_coords[..., 2] - bbox_coords[..., 0]) * (bbox_coords[..., 3] - bbox_coords[..., 1])
        best_volumes, ids = volumes.topk(5, dim=-1)

        mask_features = []
        return_mask = []

        for i, val in enumerate(ids):

            mask_valid = best_volumes[i] > 0
            valid_pairs = [(tmp, j) for tmp, j in enumerate(val) if mask_valid[tmp]]
            
            if not valid_pairs:
                mask_features.append(np.ones((CLIP_EMB_DIM,)).astype(np.float32) / np.sqrt(CLIP_EMB_DIM))
                return_mask.append(0)
                print(f'Skip mask in {scene_id}')
                continue
            

            batch_images = []
            for tmp, j in valid_pairs:
                image = images[j]
                mask_np = mask_2d[i][j].cpu().numpy()
                cropped = get_cropped_image(mask_np, image)
                batch_images.extend(cropped)
            
            if not batch_images:
                mask_features.append(np.ones((CLIP_EMB_DIM,)).astype(np.float32) / np.sqrt(CLIP_EMB_DIM))
                return_mask.append(0)
                print(f'Skip mask in {scene_id}')
                continue
            
            input_tensors = []
            for img in batch_images:
                squared = pad_into_square(Image.fromarray(img))
                input_tensors.append(preprocess(squared))
            
            input_batch = torch.stack(input_tensors).cuda()
            
            with torch.no_grad():
                features = self.model.encode_image(input_batch).float()
                features = features / features.norm(dim=-1, keepdim=True)
                mask_features.append(features.mean(0).cpu().numpy())
                return_mask.append(1)
        return mask_features, return_mask

def save_pred(masks, labels, scores, return_mask, out_path):
    masks = masks[return_mask].T.cpu().numpy()
    labels = labels[return_mask].cpu().numpy()
    scores = scores[return_mask].cpu().numpy()
    pred_dict = {
        "pred_masks": masks, 
        "pred_score": scores,
        "pred_classes": labels
    }
    np.savez(out_path, **pred_dict)

if __name__ == '__main__':
    args = get_args()
    if args.num_classes == 200:
        LABELS = SCANNET200_LABELS
    if args.num_classes == 20:
        LABELS = SCANNET20_LABELS
    if args.num_classes == 60:
        LABELS = SCANNET50_LABELS
    clip = FastCLIPFeatures(args, True)
    seq_name_list = args.seq_name_list.split('+')
    for seq_name in tqdm(seq_name_list):
        args.seq_name = seq_name
        dataset = get_dataset(args)
        masks_path = os.path.join('../data/prediction', args.config + '_class_agnostic', seq_name + '.npz')
        masks = torch.from_numpy(np.load(masks_path, allow_pickle=True)['pred_masks']).to('cuda').T
        labels, scores, return_mask = clip(dataset, masks)
        return_mask = return_mask.to(torch.bool)
        out_path = os.path.join('../data/prediction', args.config, seq_name + '.npz')
        save_pred(masks, labels, scores, return_mask, out_path)
