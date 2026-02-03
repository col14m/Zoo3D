# based on https://github.com/col14m/TUN3D/blob/main/recognition/tun3d/indoor_eval.py
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from terminaltables import AsciiTable

from utils.config import get_dataset, get_args
import pickle

def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (np.ndarray): Recalls with shape of (num_scales, num_dets)
            or (num_dets, ).
        precisions (np.ndarray): Precisions with shape of
            (num_scales, num_dets) or (num_dets, ).
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or np.ndarray: Calculated average precision.
    """
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape
    assert recalls.ndim == 2

    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    return ap


def get_iou_3d(boxes1, boxes2):
    """Calculate 3D IoU between two sets of boxes.
    
    Args:
        boxes1 (torch.Tensor): (N, 7) tensor, format: [x, y, z, l, w, h, heading]
        boxes2 (torch.Tensor): (M, 7) tensor, format: [x, y, z, l, w, h, heading]
        
    Returns:
        torch.Tensor: (N, M) IoU matrix
    """
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    
    # Extract dimensions
    l1, w1, h1 = boxes1[:, 3], boxes1[:, 4], boxes1[:, 5]
    l2, w2, h2 = boxes2[:, 3], boxes2[:, 4], boxes2[:, 5]
    
    # Calculate volumes
    vol1 = l1 * w1 * h1
    vol2 = l2 * w2 * h2
    
    x1_min = boxes1[:, 0] - l1 / 2
    x1_max = boxes1[:, 0] + l1 / 2
    y1_min = boxes1[:, 1] - w1 / 2
    y1_max = boxes1[:, 1] + w1 / 2
    z1_min = boxes1[:, 2] - h1 / 2
    z1_max = boxes1[:, 2] + h1 / 2
    
    x2_min = boxes2[:, 0] - l2 / 2
    x2_max = boxes2[:, 0] + l2 / 2
    y2_min = boxes2[:, 1] - w2 / 2
    y2_max = boxes2[:, 1] + w2 / 2
    z2_min = boxes2[:, 2] - h2 / 2
    z2_max = boxes2[:, 2] + h2 / 2
    
    # Calculate intersection
    inter_x_min = torch.max(x1_min.unsqueeze(1), x2_min.unsqueeze(0))
    inter_x_max = torch.min(x1_max.unsqueeze(1), x2_max.unsqueeze(0))
    inter_y_min = torch.max(y1_min.unsqueeze(1), y2_min.unsqueeze(0))
    inter_y_max = torch.min(y1_max.unsqueeze(1), y2_max.unsqueeze(0))
    inter_z_min = torch.max(z1_min.unsqueeze(1), z2_min.unsqueeze(0))
    inter_z_max = torch.min(z1_max.unsqueeze(1), z2_max.unsqueeze(0))
    
    inter_x = torch.clamp(inter_x_max - inter_x_min, min=0)
    inter_y = torch.clamp(inter_y_max - inter_y_min, min=0)
    inter_z = torch.clamp(inter_z_max - inter_z_min, min=0)
    
    inter_vol = inter_x * inter_y * inter_z
    
    # Calculate union
    union_vol = vol1.unsqueeze(1) + vol2.unsqueeze(0) - inter_vol
    
    # Calculate IoU
    iou = inter_vol / union_vol
    
    return iou


def eval_det_cls(pred, gt, iou_thr=None):
    """Generic functions to compute precision/recall for object detection for a
    single class.

    Args:
        pred (dict): Predictions mapping from image id to bounding boxes
            and scores.
        gt (dict): Ground truths mapping from image id to bounding boxes.
        iou_thr (list[float]): A list of iou thresholds.

    Return:
        tuple (np.ndarray, np.ndarray, float): Recalls, precisions and
            average precision.
    """

    # {img_id: {'bbox': box structure, 'det': matched list}}
    class_recs = {}
    npos = 0
    for img_id in gt.keys():
        cur_gt_num = len(gt[img_id])
        if cur_gt_num != 0:
            gt_cur = torch.zeros([cur_gt_num, 6], dtype=torch.float32)
            for i in range(cur_gt_num):
                gt_cur[i] = gt[img_id][i]
        else:
            gt_cur = torch.zeros([0, 6], dtype=torch.float32)
        det = [[False] * len(gt_cur) for i in iou_thr]
        npos += len(gt_cur)
        class_recs[img_id] = {'bbox': gt_cur, 'det': det}

    # construct dets
    image_ids = []
    confidence = []
    ious = []
    for img_id in pred.keys():
        cur_num = len(pred[img_id])
        if cur_num == 0:
            continue
        pred_cur = torch.zeros((cur_num, 6), dtype=torch.float32)
        box_idx = 0
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            pred_cur[box_idx] = box
            box_idx += 1

        if img_id in class_recs:
            gt_cur = class_recs[img_id]['bbox']
        else:
            gt_cur = torch.zeros([0, 6], dtype=torch.float32)
        if len(gt_cur) > 0:
            # calculate iou in each image
            iou_cur = get_iou_3d(pred_cur, gt_cur)
            for i in range(cur_num):
                ious.append(iou_cur[i].numpy())
        else:
            for i in range(cur_num):
                ious.append(np.zeros(1))

    confidence = np.array(confidence)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    image_ids = [image_ids[x] for x in sorted_ind]
    ious = [ious[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp_thr = [np.zeros(nd) for i in iou_thr]
    fp_thr = [np.zeros(nd) for i in iou_thr]
    for d in range(nd):
        if image_ids[d] in class_recs:
            R = class_recs[image_ids[d]]
        else:
            R = {'bbox': torch.zeros([0, 6], dtype=torch.float32)}
        iou_max = -np.inf
        BBGT = R['bbox']
        cur_iou = ious[d]

        if len(BBGT) > 0:
            # compute overlaps
            for j in range(len(BBGT)):
                iou = cur_iou[j]
                if iou > iou_max:
                    iou_max = iou
                    jmax = j

        for iou_idx, thresh in enumerate(iou_thr):
            if iou_max > thresh:
                if not R['det'][iou_idx][jmax]:
                    tp_thr[iou_idx][d] = 1.
                    R['det'][iou_idx][jmax] = 1
                else:
                    fp_thr[iou_idx][d] = 1.
            else:
                fp_thr[iou_idx][d] = 1.

    ret = []
    for iou_idx, thresh in enumerate(iou_thr):
        # compute precision recall
        fp = np.cumsum(fp_thr[iou_idx])
        tp = np.cumsum(tp_thr[iou_idx])
        recall = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = average_precision(recall, precision)
        ret.append((recall, precision, ap))

    return ret


def eval_map_recall(pred, gt, ovthresh=None):
    """Evaluate mAP and recall.

    Generic functions to compute precision/recall for object detection
        for multiple classes.

    Args:
        pred (dict): Information of detection results,
            which maps class_id and predictions.
        gt (dict): Information of ground truths, which maps class_id and
            ground truths.
        ovthresh (list[float], optional): iou threshold. Default: None.

    Return:
        tuple[dict]: dict results of recall, AP, and precision for all classes.
    """

    ret_values = {}
    for classname in gt.keys():
        if classname in pred:
            ret_values[classname] = eval_det_cls(pred[classname],
                                                 gt[classname], ovthresh)
    recall = [{} for i in ovthresh]
    precision = [{} for i in ovthresh]
    ap = [{} for i in ovthresh]

    for label in gt.keys():
        for iou_idx, thresh in enumerate(ovthresh):
            if label in pred:
                recall[iou_idx][label], precision[iou_idx][label], ap[iou_idx][
                    label] = ret_values[label][iou_idx]
            else:
                recall[iou_idx][label] = np.zeros(1)
                precision[iou_idx][label] = np.zeros(1)
                ap[iou_idx][label] = np.zeros(1)

    return recall, precision, ap


def indoor_eval(gt_annos,
                dt_annos,
                metric,
                label2cat,
                logger=None):
    """Indoor Evaluation.

    Evaluate the result of the detection.

    Args:
        gt_annos (list[dict]): Ground truth annotations.
        dt_annos (list[dict]): Detection annotations. the dict
            includes the following keys

            - labels_3d (torch.Tensor): Labels of boxes.
            - bboxes_3d (torch.Tensor): 3D bounding boxes as tensors
            - scores_3d (torch.Tensor): Scores of boxes.
        metric (list[float]): IoU thresholds for computing average precisions.
        label2cat (tuple): Map from label to category.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Return:
        dict[str, float]: Dict of results.
    """
    assert len(dt_annos) == len(gt_annos)
    pred = {}  # map {class_id: pred}
    gt = {}  # map {class_id: gt}
    for img_id in range(len(dt_annos)):
        # parse detected annotations
        det_anno = dt_annos[img_id]
        if 'labels_3d' in det_anno and len(det_anno['labels_3d']) > 0:
            labels = det_anno['labels_3d']
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
                
            bboxes = det_anno['bboxes_3d']
            if isinstance(bboxes, torch.Tensor):
                pass  # Keep as tensor
            else:
                # Convert to tensor if not already
                bboxes = torch.tensor(bboxes)
                
            scores = det_anno['scores_3d']
            if isinstance(scores, torch.Tensor):
                scores = scores.numpy()
                
            for i in range(len(labels)):
                label = int(labels[i])
                bbox = bboxes[i]
                score = float(scores[i])
                if label not in pred:
                    pred[label] = {}
                if img_id not in pred[label]:
                    pred[label][img_id] = []
                pred[label][img_id].append((bbox, score))

        # parse gt annotations
        gt_anno = gt_annos[img_id]

        if 'gt_bboxes_3d' in gt_anno:
            gt_boxes = gt_anno['gt_bboxes_3d']
            if isinstance(gt_boxes, torch.Tensor):
                pass  # Keep as tensor
            else:
                gt_boxes = torch.tensor(gt_boxes)
        else:
            gt_boxes = torch.zeros((0, 6), dtype=torch.float32)
            
        if 'gt_labels_3d' in gt_anno:
            labels_3d = gt_anno['gt_labels_3d']
            if isinstance(labels_3d, torch.Tensor):
                labels_3d = labels_3d.numpy()
        else:
            labels_3d = []

        for i in range(len(labels_3d)):
            label = int(labels_3d[i])
            bbox = gt_boxes[i]
            if label not in gt:
                gt[label] = {}
            if img_id not in gt[label]:
                gt[label][img_id] = []
            gt[label][img_id].append(bbox)

    rec, prec, ap = eval_map_recall(pred, gt, metric)
    
    ret_dict = dict()
    header = ['classes']
    table_columns = [[label2cat[label]
                      for label in ap[0].keys()] + ['Overall']]

    for i, iou_thresh in enumerate(metric):
        header.append(f'AP_{iou_thresh:.2f}')
        header.append(f'AR_{iou_thresh:.2f}')
        rec_list = []
        for label in ap[i].keys():
            ret_dict[f'{label2cat[label]}_AP_{iou_thresh:.2f}'] = float(
                ap[i][label][0])
        ret_dict[f'mAP_{iou_thresh:.2f}'] = float(
            np.nanmean(list(ap[i].values())))

        table_columns.append(list(map(float, list(ap[i].values()))))
        table_columns[-1] += [ret_dict[f'mAP_{iou_thresh:.2f}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

        for label in rec[i].keys():
            ret_dict[f'{label2cat[label]}_rec_{iou_thresh:.2f}'] = float(
                rec[i][label][-1])
            rec_list.append(rec[i][label][-1])
        ret_dict[f'mAR_{iou_thresh:.2f}'] = float(np.nanmean(rec_list))

        table_columns.append(list(map(float, rec_list)))
        table_columns[-1] += [ret_dict[f'mAR_{iou_thresh:.2f}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print('\n' + table.table)

    return ret_dict


def get_bboxes_by_masks(masks, points):
        boxes = []
        for mask in masks:
            object_points = points[mask][:, :3]
            xyz_min = object_points.min(dim=0).values
            xyz_max = object_points.max(dim=0).values
            center = (xyz_max + xyz_min) / 2
            size = xyz_max - xyz_min
            box = torch.cat((center, size))
            boxes.append(box)
        if len(boxes) == 0:
            boxes = torch.tensor(boxes)
        else:
            boxes = torch.stack(boxes)

        return boxes

def align_points(points, axis_align_matrix):
    points = points @ axis_align_matrix[:3, :3].T
    points += axis_align_matrix[:3, -1]
    return points

def calc_metrics(args, iou_thr: List[float] = [0.25, 0.5]):
    with open(args.pkl_path, 'rb') as file:
        gt_data = pickle.load(file)
    gt_annos = []
    dt_annos = []
    metainfo = gt_data['metainfo']['categories']
    metainfo = {val: key for key, val in metainfo.items()}
    for scene in gt_data['data_list']:
        seq_name = scene['lidar_points']['lidar_path'][:-4]
        dt_data = np.load(f'../data/prediction/{args.config}/{seq_name}.npz', allow_pickle=True)
        axis_align_matrix = np.array(scene['axis_align_matrix'])
        args.seq_name = seq_name
        points = get_dataset(args).get_scene_points()
        points = torch.from_numpy(align_points(points, axis_align_matrix))
        dt_boxes = get_bboxes_by_masks(torch.from_numpy(dt_data['pred_masks']).T, points)
        det_anno = {
            'bboxes_3d': dt_boxes, 
            'scores_3d': torch.from_numpy(dt_data['pred_score']),
            'labels_3d': dt_data['pred_classes']
        }
        print(dt_data['pred_classes'])
        dt_annos.append(det_anno)
        instances = scene['instances']
        gt_bboxes = [instance['bbox_3d'] for instance in instances]
        gt_labels = [instance['bbox_label_3d'] for instance in instances]
        gt_anno = {
            'gt_bboxes_3d': torch.tensor(gt_bboxes),
            'gt_labels_3d': gt_labels
        }
        gt_annos.append(gt_anno)
    
    ret_dict = indoor_eval(
            gt_annos,
            dt_annos,
            iou_thr,
            metainfo)
    return ret_dict


if __name__ == '__main__':
    args = get_args()
    calc_metrics(args)