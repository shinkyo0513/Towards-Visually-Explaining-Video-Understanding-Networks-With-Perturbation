import os

basedir = '/home/acb11711tx/lzq/ModelVisualization/test_data/kinetics'
video_names = [name[:-4] for name in os.listdir(basedir) if '.mp4' in name]
for video_name in video_names:
    video_name = video_name
    savedir = os.path.join(basedir, video_name)
    os.makedirs(savedir, exist_ok=True)

    savedir_ = savedir.replace(' ', '\ ')
    video_name_ = video_name.replace(' ', '\ ')
    print(video_name_)
    os.system(f'ffmpeg -i {video_name_}.mp4 -vf fps=10 {video_name_}/%09d.png')
    print(f'{video_name} finished!')