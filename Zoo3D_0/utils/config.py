import argparse
from dataset.scannet import ScanNetDataset
import json

def update_args(args):
    config = args.config
    config_path = f'configs/{config}.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    for key in config:
        setattr(args, key, config[key])
    return args

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_name', type=str)
    parser.add_argument('--seq_name_list', type=str)
    parser.add_argument('--config', type=str, default='scannet')
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    args = update_args(args)
    return args

def get_dataset(args):
    if args.dataset == 'scannet':
        if args.task == 'point_cloud':
            dataset = ScanNetDataset(args.seq_name, data_dir='../data/scannet/posed_images', task=args.task, num_classes=args.num_classes, frame_list=args.frame_ids)
        elif args.task == 'posed_images':
            dataset = ScanNetDataset(args.seq_name, data_dir='../data/scannet/rec_posed_images', task=args.task, num_classes=args.num_classes, frame_list=args.frame_ids)
        elif args.task == 'unposed_images':
            dataset = ScanNetDataset(args.seq_name, data_dir='../data/scannet/rec_unposed_images', task=args.task, num_classes=args.num_classes, frame_list=args.frame_ids)
        else:
            print(args.dataset, args.task)
            raise NotImplementedError
    else:
        print(args.dataset)
        raise NotImplementedError
    return dataset

