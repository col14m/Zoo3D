import os
from tqdm import tqdm
import time
from utils.config import get_args

CUDA_LIST = [0]

def execute_commands(commands_list, command_type, process_num):
    print('====> Start', command_type)
    from multiprocessing import Pool
    pool = Pool(process_num)
    for _ in tqdm(pool.imap_unordered(os.system, commands_list), total=len(commands_list)):
        pass
    pool.close()
    pool.join()
    pool.terminate()
    print('====> Finish', command_type)

def get_seq_name_list(dataset):
    if dataset == 'scannet':
        file_path = 'splits/scannet.txt'
    elif dataset == 'scannetpp':
        file_path = 'splits/scannetpp.txt'
    with open(file_path, 'r') as f:
        seq_name_list = f.readlines()
    seq_name_list = [seq_name.strip() for seq_name in seq_name_list]
    return seq_name_list

def parallel_compute(general_command, command_name, resource_type, cuda_list, seq_name_list):
    cuda_num = len(cuda_list)
    
    if resource_type == 'cuda':
        commands = []
        for i, cuda_id in enumerate(cuda_list):
            process_seq_name = seq_name_list[i::cuda_num]
            if len(process_seq_name) == 0:
                continue
            process_seq_name = '+'.join(process_seq_name)
            command = f'CUDA_VISIBLE_DEVICES={cuda_id} {general_command % process_seq_name}'
            commands.append(command)
        execute_commands(commands, command_name, cuda_num)
    elif resource_type == 'cpu':
        commands = []
        for seq_name in seq_name_list:
            commands.append(f'{general_command} --seq_name {seq_name}')
        execute_commands(commands, command_name, cuda_num)

def get_label_text_feature(cuda_id):
    command = f'CUDA_VISIBLE_DEVICES={cuda_id} python -m semantics.extract_label_features'
    os.system(command)

def main(args):
    dataset = args.dataset
    task = args.task
    config = args.config
    cropformer_path = args.cropformer_path

    if dataset == 'scannet':
        if task == 'point_cloud':
            root = '../data/scannet/posed_images'
        elif task == 'posed_images':
            root = '../data/scannet/rec_posed_images'
        if task == 'unposed_images':
            root = '../data/scannet/rec_unposed_images'
        image_path_pattern = '*.jpg'
    
    seq_name_list = get_seq_name_list(dataset)

    t0 = time.time()
    
    print('There are %d scenes' % len(seq_name_list))
    
    # Step 1: use Cropformer to get 2D instance masks for all sequences.
    parallel_compute(f'python third_party/detectron2/projects/CropFormer/demo_cropformer/mask_predict.py --config-file third_party/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml --root {root} --image_path_pattern {image_path_pattern} --dataset {args.dataset} --task {task} --seq_name_list %s --opts MODEL.WEIGHTS {cropformer_path}', 'predict mask', 'cuda', CUDA_LIST, seq_name_list)

    # # Step 2: Mask clustering using our proposed method.
    parallel_compute(f'python main.py --config {config} --seq_name_list %s', 'mask clustering', 'cuda', CUDA_LIST, seq_name_list)
    
    parallel_compute(f'python -m semantics.points_proj_precompute --config {config} --seq_name_list %s', 'project points on frames', 'cuda', CUDA_LIST, seq_name_list)

    get_label_text_feature(CUDA_LIST[0])

    parallel_compute(f'python -m semantics.open_voc --config {config} --seq_name_list %s', 'get open-vocab preds', 'cuda', CUDA_LIST, seq_name_list)
    
    os.system(f'python -m evaluation.metric --config {config}')

    print('total time', (time.time() - t0)//60, 'min')
    print('Average time', (time.time() - t0) / len(seq_name_list), 'sec')

if __name__ == '__main__':
    args = get_args()
    main(args)