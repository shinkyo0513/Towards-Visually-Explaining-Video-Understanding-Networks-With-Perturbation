import argparse
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader

import os
import time
import copy
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import csv
import json

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(".")
sys.path.append("..")

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root

from utils.ImageShow import *

from visual_meth.integrated_grad import integrated_grad
from visual_meth.gradients import gradients
from visual_meth.perturbation import video_perturbation
from visual_meth.grad_cam import grad_cam

parser = argparse.ArgumentParser()
parser.add_argument("--videos_dir", type=str, default='')
parser.add_argument("--model", type=str, default='r2plus1d',
                    choices=['r2plus1d', 'r3d', 'mc3', 'i3d', 'tsn', 'trn', 'tsm'])
parser.add_argument("--pretrain_dataset", type=str, default='kinetics',
                    choices=['', 'kinetics', 'epic-kitchens-verb', 'epic-kitchens-noun'])
parser.add_argument("--vis_method", type=str, default='integrated_grad',
                    choices=['grad', 'grad*input', 'integrated_grad', 'grad_cam', 'perturb'])
parser.add_argument("--save_label", type=str, default='')
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--num_iter", type=int, default=2000)
parser.add_argument('--perturb_area', type=float, default=0.1,
                    choices=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2])
parser.add_argument('--polarity', type=str, default='positive',
                    choices=['positive', 'negative'])
args = parser.parse_args()

# assert args.num_gpu >= -1
# if args.num_gpu == 0:
#     num_devices = 0
#     multi_gpu = False
#     device = torch.device("cpu")
# elif args.num_gpu == 1:
#     num_devices = 1
#     multi_gpu = False
#     device = torch.device("cuda")
# elif args.num_gpu == -1:
#     num_devices = torch.cuda.device_count()
#     multi_gpu = (num_devices > 1)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# else:
#     num_devices = args.num_gpu
#     assert torch.cuda.device_count() >= num_devices, \
#         f'Assign {args.num_gpu} GPUs, but only detected only {torch.cuda.device_count()} GPUs. Exiting...'
#     multi_gpu = True
#     device = torch.device("cuda")

if args.no_gpu:
    device = torch.device("cpu")
    num_devices = 0
else:
    device = torch.device("cuda")
    num_devices = 1

assert os.path.isdir(args.videos_dir), \
    f'Given directory of data does not exist: {args.videos_dir}.'

if args.pretrain_dataset == 'kinetics':
    if args.model == 'i3d':
        from model_def.i3d import I3D as model
        model_ft = model(num_classes=400)
        i3d_pt_dir = os.path.join(proj_root, 'model_param/kinetics400_rgb_i3d.pth')
        model_ft.load_state_dict(torch.load(i3d_pt_dir))
        clip_length = 16
    elif args.model == 'tsm':
        from model_def.tsm import tsm as model
        model_ft = model(400, segment_count=8, pretrained=args.pretrain_dataset)
        clip_length = 8
    else:   # Load pretrained models from PyTorch directly
        clip_length = 16
        if args.model == 'r2plus1d':
            from torchvision.models.video import r2plus1d_18 as model
        elif args.model == 'mc3':
            from torchvision.models.video import mc3_18 as model
        elif args.model == 'r3d':
            from torchvision.models.video import r3d_18 as model
        else:
            raise Exception(f'Given model of {args.model} has no pretrain on {args.pretrain_dataset}.')
        model_ft = model(pretrained=True)

    model_ft = model_ft.to(device)
    model_ft.eval()
    # if multi_gpu:
    #     model_ft = nn.DataParallel(model_ft, device_ids=list(range(num_devices)))

    kinetics400_classes = os.path.join(proj_root, 'test_data/kinetics/classes.json')
    class_namelist = json.load(open(kinetics400_classes))

elif 'epic-kitchens' in args.pretrain_dataset:
    if 'noun' in args.pretrain_dataset:
        epic_classes = os.path.join(proj_root, 'test_data/epic-kitchens-noun/EPIC_noun_classes.csv')
    elif 'verb' in args.pretrain_dataset:
        epic_classes = os.path.join(proj_root, 'test_data/epic-kitchens-verb/EPIC_verb_classes.csv')
    else:
        raise Exception(f'EPIC-Kitchens only supports two sub-tasks (noun & verb), given {args.pretrain_dataset}.')
    class_namelist = [row['class_key'] for ridx, row in pd.read_csv(epic_classes).iterrows()]
    class_num = len(class_namelist)

    if args.model == 'tsm':
        from model_def.tsm import tsm as model
        model_ft = model(class_num, segment_count=8, pretrained=args.pretrain_dataset)
        clip_length = 8
    else:
        raise Exception(f'{args.pretrain_dataset} has only pretrained TSM model. Given {args.model}.')
    model_ft = model_ft.to(device)
    model_ft.eval()

from datasets.universal_dataset import UniversalDataset as dataset
test_dataset = dataset(args.videos_dir, args.model, class_namelist, clip_length=clip_length)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print(f'Num of test samples:{len(test_dataset)}')

for sample in tqdm(test_dataloader):
    inp = sample[0].to(device)
    label = sample[1].to(dtype=torch.long)

    inp_np = voxel_tensor_to_np(inp[0].detach().cpu())   # 3 x num_f x 224 224

    if args.vis_method == 'integrated_grad':
        res = integrated_grad(inp, label, model_ft, device, steps=50, polarity=args.polarity)
        heatmap_np = res[0].numpy()
    elif args.vis_method == 'grad':
        res = gradients(inp, label, model_ft, device, polarity=args.polarity)
        heatmap_np = res[0].numpy()
    elif args.vis_method == 'grad*input':
        res = gradients(inp, label, model_ft, device, multiply_input=True, polarity=args.polarity)
        heatmap_np = res[0].numpy()
    elif args.vis_method == 'grad_cam':
        if args.model in ['i3d']:
            layer_name = ['mixed_5c']
        elif args.model in ['r2plus1d', 'mc3', 'r3d']:   # Load pretrained models from PyTorch directly
            layer_name = ['layer4']
        # elif args.model in ['tsm', 'tsn']:
        #     layer_name = ['model', 'base_model', 'layer4']
        else:
            raise Exception(f'Grad-CAM does not support {args.model} currently')
        res = grad_cam(inp, label, model_ft, device, layer_name=layer_name, norm_vis=True)
        heatmap_np = overlap_maps_on_voxel_np(inp_np, res[0,0].cpu().numpy(), norm_map=False)
    elif args.vis_method == 'perturb':
        sigma = 11 if inp.shape[-1] == 112 else 23
        res = video_perturbation(
                    model_ft, inp, label, areas=[args.perturb_area], sigma=sigma, 
                    max_iter=args.num_iter, variant="preserve",
                    num_devices=num_devices, print_iter=100, perturb_type="fade")[0]
        heatmap_np = overlap_maps_on_voxel_np(inp_np, res[0,0].cpu().numpy(), norm_map=False)

    sample_name = sample[2][0].split("/")[-1]
    plot_save_name = f"{sample_name}.png"
    if args.vis_method in ['grad', 'grad*input', 'integrated_grad']:
        plot_save_name = plot_save_name.replace('.png', f'{args.polarity}.png')
    plot_save_dir = os.path.join(proj_root, "visual_res", args.vis_method, args.model)
    if args.save_label != '':
        plot_save_dir = os.path.join(plot_save_dir, args.save_label)
    os.makedirs(plot_save_dir, exist_ok=True)

    show_txt = f"{sample_name}"
    plot_voxel_np(inp_np, heatmap_np, title=show_txt, 
                    save_path=os.path.join(plot_save_dir, plot_save_name) )


    

