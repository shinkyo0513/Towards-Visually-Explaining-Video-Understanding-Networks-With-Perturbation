import torch
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image
from skimage import transform, filters
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def pil_to_tensor(pil_image):
    r"""Convert a PIL image to a tensor.
    Args:
        pil_image (:class:`PIL.Image`): PIL image.
    Returns:
        :class:`torch.Tensor`: the image as a :math:`3\times H\times W` tensor
        in the [0, 1] range.
    """
    pil_image = np.array(pil_image)
    if len(pil_image.shape) == 2:
        pil_image = pil_image[:, :, None]
    return torch.tensor(pil_image, dtype=torch.float32).permute(2, 0, 1) / 255

def img_tensor_to_pil(img_tensor):
    lim = [img_tensor.min(), img_tensor.max()]
    img_tensor = img_tensor - lim[0]  # also makes a copy
    img_tensor.mul_(1 / (lim[1] - lim[0]))
    img_tensor = torch.clamp(img_tensor, min=0, max=1)
    img_pil = TF.to_pil_image(img_tensor)
    return img_pil

def img_tensor_to_np(img_tensor):
    lim = [img_tensor.min(), img_tensor.max()]
    img_tensor = img_tensor - lim[0]  # also makes a copy
    img_tensor.mul_(1 / (lim[1] - lim[0]))
    img_tensor = torch.clamp(img_tensor, min=0, max=1)
    img_np = img_tensor.numpy()
    return img_np

def voxel_tensor_to_np(voxel_tensor):
    # voxel_tensor: CxTxHxW
    voxel_np = []
    for t in range(voxel_tensor.shape[1]):
        img_np = img_tensor_to_np(voxel_tensor[:,t,:,:])
        voxel_np.append(img_np)
    voxel_np = np.stack(voxel_np, axis=1)
    return voxel_np

def map_to_colormap(attMap, resize=(), norm_map=False, blur=False):
    attMap = attMap.copy()
    if norm_map:
        attMap -= attMap.min()
        if attMap.max() > 0:
            attMap /= attMap.max()

    if resize != ():
        attMap = transform.resize(attMap, resize, order=3)

    if blur:
        attMap = filters.gaussian(attMap, 0.02*max(resize))
        attMap -= attMap.min()
        attMap /= attMap.max()

    cmap = plt.get_cmap('jet')
    colormap = cmap(attMap)
    colormap = np.delete(colormap, 3, 2)
    colormap = np.transpose(colormap, (2,0,1))    #3x224x224
    return attMap, colormap

def overlap_map_on_img(img_np, attMap, norm_map=False, blur=False):
    # img_np: CxHxW, attMap: h x w
    plt.axis('off')
    resized_map, colormap = map_to_colormap(attMap, resize=(img_np.shape[1:]), norm_map=False, blur=blur)
    resized_map = 1*(1-resized_map**0.8)*img_np + (resized_map**0.8)*colormap
    return resized_map

def overlap_maps_on_voxel_np(voxel_np, attMaps, norm_map=False, blur=False):
    overlaps = []
    for t in range(voxel_np.shape[1]):
        img_np = voxel_np[:,t,:,:]
        attMap = attMaps[t,:,:]
        overlap = overlap_map_on_img(img_np, attMap, norm_map, blur)
        overlaps.append(overlap)
    overlaps = np.stack(overlaps, axis=1)    # 3x16x224x224
    return overlaps

def img_np_show (img_np, interpolation='lanczos'):
    bitmap = np.transpose(img_np, (1,2,0))  # HxWxC
    handle = plt.imshow(
        bitmap, interpolation=interpolation, vmin=0, vmax=1)
    curr_ax = plt.gca()
    curr_ax.axis('off')
    return handle

def imsc(img, *args, lim=None, quiet=False, interpolation='lanczos', **kwargs):
    r"""Rescale and displays an image represented as a img.
    The function scales the img :attr:`im` to the [0 ,1] range.
    The img is assumed to have shape :math:`3\times H\times W` (RGB)
    :math:`1\times H\times W` (grayscale).
    Args:
        img (:class:`torch.Tensor` or :class:`PIL.Image`): image.
        quiet (bool, optional): if False, do not display image.
            Default: ``False``.
        lim (list, optional): maximum and minimum intensity value for
            rescaling. Default: ``None``.
        interpolation (str, optional): The interpolation mode to use with
            :func:`matplotlib.pyplot.imshow` (e.g. ``'lanczos'`` or
            ``'nearest'``). Default: ``'lanczos'``.
    Returns:
        :class:`torch.Tensor`: Rescaled image img.
    """
    if isinstance(img, Image.Image):
        img = pil_to_tensor(img)
    handle = None
    with torch.no_grad():
        if not lim:
            lim = [img.min(), img.max()]
        img = img - lim[0]  # also makes a copy
        img.mul_(1 / (lim[1] - lim[0]))
        img = torch.clamp(img, min=0, max=1)
        if not quiet:
            bitmap = img.expand(3, *img.shape[1:]).permute(1, 2, 0).cpu().numpy()
            handle = plt.imshow(
                bitmap, *args, interpolation=interpolation, **kwargs)
            curr_ax = plt.gca()
            curr_ax.axis('off')
    return img, handle

def plot_voxel(voxel, saliency, show_plot=False, save_path=None):
    # batch_size = len(input)
    num_frame = voxel.shape[1]
    num_row = 2 * num_frame//8 

    plt.clf()
    fig = plt.figure(figsize=(16,num_row*2))
    for i in range(num_frame):
        plt.subplot(num_row, 8, (i//8)*16+i%8+1)
        imsc(voxel[:,i,:,:])
        plt.title(i, fontsize=8)

        plt.subplot(num_row, 8, (i//8)*16+i%8+8+1)
        imsc(saliency[:,i,:,:], interpolation='none')

    # Save figure if path is specified.
    if save_path:
        save_dir = os.path.dirname(os.path.abspath(save_path))
        # Create directory if necessary.
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ext = os.path.splitext(save_path)[1].strip('.')
        plt.savefig(save_path, format=ext, bbox_inches='tight')

    # Show plot if desired.
    if show_plot:
        plt.show()

def plot_voxel_wbbox(voxel, saliency, bbox_tensor, 
                    show_plot=False, save_path=None):
    # batch_size = len(input)
    num_frame = voxel.shape[1]
    num_row = 2 * num_frame//8 

    for idx in range(num_frame):
        x0, y0, x1, y1 = bbox_tensor[idx,:].tolist()
        voxel[1, idx, y0:y1+1, x0] = 1.0
        voxel[1, idx, y0:y1+1, x1] = 1.0
        voxel[1, idx, y0, x0:x1+1] = 1.0
        voxel[1, idx, y1, x0:x1+1] = 1.0

    plt.clf()
    fig = plt.figure(figsize=(16,num_row*2))
    for i in range(num_frame):
        plt.subplot(num_row, 8, (i//8)*16+i%8+1)
        imsc(voxel[:,i,:,:])
        plt.title(i, fontsize=8)

        plt.subplot(num_row, 8, (i//8)*16+i%8+8+1)
        imsc(saliency[:,i,:,:], interpolation='none')

    # Save figure if path is specified.
    if save_path:
        save_dir = os.path.dirname(os.path.abspath(save_path))
        # Create directory if necessary.
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ext = os.path.splitext(save_path)[1].strip('.')
        plt.savefig(save_path, format=ext, bbox_inches='tight')

    # Show plot if desired.
    if show_plot:
        plt.show()

def plot_voxel_np(voxel_np, saliency_np, title=None, 
            show_plot=False, save_path=None):
    # batch_size = len(input)
    num_frame = voxel_np.shape[1]
    num_row = 2 * num_frame//8 

    plt.clf()
    fig = plt.figure(figsize=(16,num_row*2))
    for i in range(num_frame):
        plt.subplot(num_row, 8, (i//8)*16+i%8+1)
        img_np_show(voxel_np[:,i,:,:])
        plt.title(i, fontsize=8)

        plt.subplot(num_row, 8, (i//8)*16+i%8+8+1)
        img_np_show(saliency_np[:,i,:,:], interpolation='none')
    # fig.close()

    if title is not None:
        fig.suptitle(title, fontsize=14)

    # Save figure if path is specified.
    if save_path:
        save_dir = os.path.dirname(os.path.abspath(save_path))
        # Create directory if necessary.
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ext = os.path.splitext(save_path)[1].strip('.')
        plt.savefig(save_path, format=ext, bbox_inches='tight')

    # Show plot if desired.
    if show_plot:
        plt.show()

def plot_voxel_wbbox_np(voxel_np, saliency_np, bbox_tensor, title=None,
                    show_plot=False, save_path=None):
    # batch_size = len(input)
    num_frame = voxel_np.shape[1]
    num_row = 2 * num_frame//8 

    for idx in range(num_frame):
        x0, y0, x1, y1 = bbox_tensor[idx,:].tolist()
        voxel_np[1, idx, y0:y1+1, x0] = 1.0
        voxel_np[1, idx, y0:y1+1, x1] = 1.0
        voxel_np[1, idx, y0, x0:x1+1] = 1.0
        voxel_np[1, idx, y1, x0:x1+1] = 1.0

    plt.clf()
    fig = plt.figure(figsize=(16,num_row*2))
    for i in range(num_frame):
        plt.subplot(num_row, 8, (i//8)*16+i%8+1)
        img_np_show(voxel_np[:,i,:,:])
        plt.title(i, fontsize=8)

        plt.subplot(num_row, 8, (i//8)*16+i%8+8+1)
        img_np_show(saliency_np[:,i,:,:], interpolation='none')

    if title is not None:
        fig.suptitle(title, fontsize=14)

    # Save figure if path is specified.
    if save_path:
        save_dir = os.path.dirname(os.path.abspath(save_path))
        # Create directory if necessary.
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ext = os.path.splitext(save_path)[1].strip('.')
        plt.savefig(save_path, format=ext, bbox_inches='tight')

    # Show plot if desired.
    if show_plot:
        plt.show()

    plt.close(fig)