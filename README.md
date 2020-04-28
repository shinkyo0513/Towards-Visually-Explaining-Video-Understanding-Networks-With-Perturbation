# VideoVisual

This is a PyTorch demo implemented several visualization methods for video classification networks. The target is to provide a toolkit (as [TorchRay](https://github.com/facebookresearch/TorchRay) to image) to interprete commonly utilized video classfication networks, such as I3D, R(2+1)D, TSM et al., which is also called *attribution* task, namely the problem of determining which part of the input video is responsible for the value computed by a neural network.

The current version supports attribution methods and video classification models as following:

#### Video classification models:
* **Pretrained on Kinetics-400**: I3D, R(2+1)D, R3D, MC3, TSM;
* **Pretrained on EPIC-Kitchens**: (noun & verb): TSM.

#### Attribution methods:
* **Backprop-based**: Gradients, Gradients x Inputs, Integrated Gradients;
* **Activation-based**: GradCAM (does not support TSM now);
* **Perturbation-based**: Extremal Perturbation and Spatiotemporal Perturbation (An extension version of extremal perturbation on video inputs).

## Requirements

* Python 3.6.5 or greater
* PyTorch 1.2.0 or greater
* matplotlib==2.2.3
* numpy==1.14.3
* opencv_python==4.1.2.30
* torchvision==0.4.0a0
* torchray==1.0.0.2
* tqdm==4.45.0
* pandas==0.23.3
* scikit_image==0.15.0
* Pillow==7.1.2
* scikit_learn==0.22.2.post1

## Running the code

### Examples

#### Saptiotemporal Perturbation + I3D (pretrained on Kinetics-400)
`$ python main.py --videos_dir /home/acb11711tx/lzq/VideoVisual/test_data/kinetics/sampled_frames --model i3d --pretrain_dataset kinetics --vis_method perturb --num_iter 2000 --perturb_area 0.1`

#### Spatiotemporal Perturbation + TSM (pretrained on EPIC-Kitchens-noun)
`$ python main.py --videos_dir /home/acb11711tx/lzq/VideoVisual/test_data/epic-kitchens-noun/sampled_frames --model tsm --pretrain_dataset epic-kitchens-noun --vis_method perturb --num_iter 2000 --perturb_area 0.05`

#### Integrated Gradients + R(2+1)D (pretrained on Kinetics-400)
`$ python main.py --videos_dir /home/acb11711tx/lzq/VideoVisual/test_data/kinetics/sampled_frames --model r2plus1d --pretrain_dataset kinetics --vis_method integrated_grad`

* Outputs: The results will be defaultly saved to the directory ./visual_res/$vis_method$/$model$/$save_label$/.

## Results
![Kinectis-400 (GT = ironing)](figures/res_fig_kinetics.png)

![EPIC-Kitchens-Noun (GT = cupboard)](figures/res_fig_epic.png)

## License

TorchRay is CC-BY-NC licensed, as found in the [LICENSE](LICENSE) file.
