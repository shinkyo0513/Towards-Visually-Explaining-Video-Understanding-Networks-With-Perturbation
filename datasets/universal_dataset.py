import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import os
import json
import numpy as np 
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append(".")
sys.path.append("..")
from utils.LongRangeSample import long_range_sample

class UniversalDataset (Dataset):
    def __init__ (self, data_dir, model_name, class_namelist, clip_length=16):
        self.data_dir = data_dir
        self.model_name = model_name
        self.class_namelist = class_namelist
        self.clip_length = clip_length
        
        self.video_names = sorted(os.listdir(data_dir))
        assert len(self.video_names) > 0, f'Given directory contains no video.'

        if model_name == 'i3d':
            self.transform = transforms.Compose([
                    transforms.Resize((344, 256)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.43216, 0.39467, 0.37645], [0.22803, 0.22145, 0.21699]),
                ])
        elif model_name in ['r2plus1d', 'r3d', 'mc3']:
            self.transform = transforms.Compose([
                    transforms.Resize((172, 128)),
                    transforms.CenterCrop((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.43216, 0.39467, 0.37645], [0.22803, 0.22145, 0.21699]),
                ])
        elif model_name in ['tsm', 'tsn']:  # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
            self.transform = transforms.Compose([
                    transforms.Resize((344, 256)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__ (self):
        return len(self.video_names)

    def __getitem__ (self, idx):
        video_name = self.video_names[idx]
        if '.mp4' not in video_name:
            video_frames_dir = os.path.join(self.data_dir, video_name)
            frame_names = sorted([f for f in os.listdir(video_frames_dir) if '.png' in f or '.jpg' in f])
            num_frame = len(frame_names)
            assert num_frame > self.clip_length, \
                f"Number of frames should be larger than {self.clip_length}, given {num_frame}"

            clip_fidxs = long_range_sample(num_frame, self.clip_length, 'first')
            clip_fidxs_tensor = torch.tensor(clip_fidxs).long()

            clip_frames = [Image.open(os.path.join(video_frames_dir, 
                                f'{fidx+1:09d}.png')) for fidx in clip_fidxs]
            clip_tensor = torch.stack([self.transform(frame) for frame in clip_frames], dim=1)

            label_name = video_name.split('-')[0]
            label_idx = self.class_namelist.index(label_name)
            return clip_tensor, label_idx, video_name, clip_fidxs_tensor
        else:
            raise Exception('Cannot process MP4 file yet.')
            
        
