import os
from PIL import Image
import cv2

# basedir = '/home/acb11711tx/lzq/ModelVisualization/test_data/epic-kitchens-noun/frames'

# # for video_name in os.listdir(basedir):
# #     video_dir = os.path.join(basedir, video_name)
# #     frame_names = sorted(os.listdir(video_dir))
# #     for frame_index, frame_name in enumerate(frame_names):
# #         frame_index = int(frame_name[6:-4])
# #         os.system(f'mv {video_dir}/{frame_name} {video_dir}/{frame_index+1:09d}.jpg')
# #     print(f'{video_name} is finished.')

# for video_name in os.listdir(basedir):
#     video_dir = os.path.join(basedir, video_name)
#     frame_names = sorted(os.listdir(video_dir))
#     num_frame = len(frame_names)
#     for frame_index in range(num_frame):
#         img = Image.open(f'{video_dir}/{frame_index+1:09d}.jpg')
#         img.save(f'{video_dir}/{frame_index+1:09d}.png')
#         os.system(f'rm {video_dir}/{frame_index+1:09d}.jpg')
#     print(f'{video_name} is finished.')

# import os
import sys
sys.path.append(".")
sys.path.append("..")
from utils.LongRangeSample import long_range_sample

basedir = '/home/acb11711tx/lzq/VideoVisual/test_data'
dataset_names = ['epic-kitchens-noun', 'epic-kitchens-verb', 'kinetics']
for dataset in dataset_names:
    ds_path = os.path.join(basedir, dataset, 'frames')
    video_names = sorted(os.listdir(ds_path))
    for video_name in video_names:
        video_dir = os.path.join(ds_path, video_name)
        frame_names = sorted(os.listdir(video_dir))
        num_frame = len(frame_names)
        sampled_fidxs = long_range_sample(num_frame, 16, 'first')

        new_video_dir = video_dir.replace('frames', 'sampled_frames')
        os.makedirs(new_video_dir, exist_ok=True)
        for idx, fidx in enumerate(sampled_fidxs):
            img = Image.open(f'{video_dir}/{fidx+1:09d}.png')
            img = img.resize((344, 256))
            img.save(f'{new_video_dir}/{idx+1:09d}.png')
        print(f'{video_dir} finished!')

    