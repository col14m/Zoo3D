import torch
from utils.config import get_dataset, get_args
from utils.post_process import post_process
from graph.construction import mask_graph_construction
from graph.iterative_clustering import iterative_clustering
from tqdm import tqdm
import os

def main(args):
    dataset = get_dataset(args)
    scene_points = dataset.get_scene_points()
    frame_list = dataset.get_frame_list()

    with torch.no_grad():
        nodes, observer_num_thresholds, mask_point_clouds, point_frame_matrix = mask_graph_construction(args, scene_points, frame_list, dataset)

        object_list = iterative_clustering(nodes, observer_num_thresholds, args.view_consensus_threshold, args.debug)

        post_process(dataset, object_list, mask_point_clouds, scene_points, point_frame_matrix, frame_list, args)

if __name__ == '__main__':
    args = get_args()
    seq_name_list = args.seq_name_list.split('+')
    bad_scenes_list = []
    for seq_name in tqdm(seq_name_list):
        args.seq_name = seq_name
        try:
            main(args)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            bad_scenes_list.append(seq_name)
            continue
    print(f'Total {len(bad_scenes_list)} scenes skipped with names: {bad_scenes_list}')