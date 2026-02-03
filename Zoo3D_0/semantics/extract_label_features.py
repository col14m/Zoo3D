import open_clip
from open_clip import tokenizer
import torch
import numpy as np
from evaluation.constants import SCANNET200_LABELS, SCANNET20_LABELS, SCANNET50_LABELS
import os

def load_clip():
    print(f'[INFO] loading CLIP model...')
    # model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    open_clip_model = 'ViT-H-14-quickgelu'
    open_clip_pretrained = 'dfn5b'
    model, _, preprocess = open_clip.create_model_and_transforms(open_clip_model, pretrained=open_clip_pretrained)
    model.cuda()
    model.eval()
    print(f'[INFO]', ' finish loading CLIP model...')
    return model

def extract_text_feature(save_path, descriptions):
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    text_tokens = tokenizer.tokenize(descriptions).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()

    text_features_dict = {}
    for i, description in enumerate(descriptions):
        text_features_dict[description] = text_features[i]

    np.save(save_path, text_features_dict)

model = load_clip()
extract_text_feature('../data/text_features/scannet200.npy', SCANNET200_LABELS)
extract_text_feature('../data/text_features/scannet60.npy', SCANNET50_LABELS)
extract_text_feature('../data/text_features/scannet20.npy', SCANNET20_LABELS)