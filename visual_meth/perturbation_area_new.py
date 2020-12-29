import math
import matplotlib.pyplot as plt
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append(".")
sys.path.append("..")
from utils.CalAcc import process_activations
from utils.ImageShow import *

from torchray.utils import imsmooth, imsc
from torchray.attribution.common import resize_saliency

BLUR_PERTURBATION = "blur"
FADE_PERTURBATION = "fade"

PRESERVE_VARIANT = "preserve"
DELETE_VARIANT = "delete"
DUAL_VARIANT = "dual"

def simple_log_reward(activation, target, variant):
    N = target.shape[0]
    bs = activation.shape[0]
    b_repeat = int( bs // N )
    device = activation.device

    col_idx = target.repeat(b_repeat) # batch_size
    row_idx = torch.arange(activation.shape[0], dtype=torch.long, device=device)   # batch_size
    prob = activation[row_idx, col_idx] # batch_size

    if variant == DELETE_VARIANT:
        reward = -torch.log(1-prob)
    elif variant == PRESERVE_VARIANT:
        reward = -torch.log(prob)
    elif variant == DUAL_VARIANT:
        reward = (-torch.log(1-prob[N:])) + (-torch.log(prob[:N]))
    else:
        assert False
    return reward

class MaskGenerator:
    def __init__(self, shape, step, sigma, clamp=True, pooling_method='softmax'):
        self.shape = shape
        self.step = step
        self.sigma = sigma
        self.coldness = 20
        self.clamp = clamp
        self.pooling_method = pooling_method

        assert int(step) == step

        # self.kernel = lambda z: (z < 1).float()
        self.kernel = lambda z: torch.exp(-2 * ((z - .5).clamp(min=0)**2))

        self.margin = self.sigma
        self.padding = 1 + math.ceil((self.margin + sigma) / step)
        self.radius = 1 + math.ceil(sigma / step)
        self.shape_in = [math.ceil(z / step) for z in self.shape]
        self.shape_mid = [
            z + 2 * self.padding - (2 * self.radius + 1) + 1
            for z in self.shape_in
        ]
        self.shape_up = [self.step * z for z in self.shape_mid]
        self.shape_out = [z - step + 1 for z in self.shape_up]

        step_inv = [
            torch.tensor(zm, dtype=torch.float32) /
            torch.tensor(zo, dtype=torch.float32)
            for zm, zo in zip(self.shape_mid, self.shape_up)
        ]

        # Generate kernel weights for smoothing mask_in
        self.weight = torch.zeros((
            1,
            (2 * self.radius + 1)**2,
            self.shape_out[0],
            self.shape_out[1]
        ))

        for ky in range(2 * self.radius + 1):
            for kx in range(2 * self.radius + 1):
                uy, ux = torch.meshgrid(
                    torch.arange(self.shape_out[0], dtype=torch.float32),
                    torch.arange(self.shape_out[1], dtype=torch.float32)
                )
                iy = torch.floor(step_inv[0] * uy) + ky - self.padding
                ix = torch.floor(step_inv[1] * ux) + kx - self.padding

                delta = torch.sqrt(
                    (uy - (self.margin + self.step * iy))**2 +
                    (ux - (self.margin + self.step * ix))**2
                )

                k = ky * (2 * self.radius + 1) + kx

                self.weight[0, k] = self.kernel(delta / sigma)

    def generate(self, mask_in):
        # mask_in: Nx1xHxW --> mask: Nx1xS_outxS_out
        mask = F.unfold(mask_in,
                        (2 * self.radius + 1,) * 2,
                        padding=(self.padding,) * 2)
        mask = mask.reshape(
            mask_in.shape[0], -1, self.shape_mid[0], self.shape_mid[1])
        mask = F.interpolate(mask, size=self.shape_up, mode='nearest')
        mask = F.pad(mask, (0, -self.step + 1, 0, -self.step + 1))
        mask = self.weight * mask

        if self.pooling_method == 'sigmoid':
            if self.coldness == float('+Inf'):
                mask = (mask.sum(dim=1, keepdim=True) - 5 > 0).float()
            else:
                mask = torch.sigmoid(
                    self.coldness * mask.sum(dim=1, keepdim=True) - 3
                )
        elif self.pooling_method == 'softmax':
            if self.coldness == float('+Inf'):  # max normalization
                mask = mask.max(dim=1, keepdim=True)[0]
            else:   # smax normalization
                mask = (
                    mask * F.softmax(self.coldness * mask, dim=1)
                ).sum(dim=1, keepdim=True)
        elif self.pooling_method == 'sum':
            mask = mask.sum(dim=1, keepdim=True)
        else:
            assert False, f"Unknown pooling method {self.pooling_method}"

        m = round(self.margin)
        if self.clamp:
            mask = mask.clamp(min=0, max=1)
        cropped = mask[:, :, m:m + self.shape[0], m:m + self.shape[1]]
        return cropped, mask

    def to(self, dev):
        """Switch to another device.
        Args:
            dev: PyTorch device.
        Returns:
            MaskGenerator: self.
        """
        self.weight = self.weight.to(dev)
        return self

class Perturbation:
    def __init__(self, input, num_levels=8, max_blur=20, type=BLUR_PERTURBATION):
        self.type = type
        self.num_levels = num_levels
        self.pyramid = []
        assert num_levels >= 2
        assert max_blur > 0
        with torch.no_grad():
            for sigma in torch.linspace(0, 1, self.num_levels):
                if type == BLUR_PERTURBATION:
                    # input could be a batched tensor with size of NxCxHxW
                    y = imsmooth(input, sigma=(1 - sigma) * max_blur)
                    # ouput y has size of NxCxHxW
                elif type == FADE_PERTURBATION:
                    y = input * sigma
                else:
                    assert False
                self.pyramid.append(y)
            # self.pyramid = torch.cat(self.pyramid, dim=0)
            self.pyramid = torch.stack(self.pyramid, dim=1) # NxLxCxHxW, L=num_levels

    def apply(self, mask):
        # mask: A*N*T x1xHxW
        n = mask.shape[0]           # n = A*N*T
        inp_n = self.pyramid.shape[0]    # inp_n = N*T
        num_area = int(n / inp_n)   # A
        # starred expression: unpack a list to separated numbers
        w = mask.reshape(n, 1, *mask.shape[1:]) # A*N*T x1x1xHxW, mask.unsqueeze(1)
        w = w * (self.num_levels - 1)   # w = 7*w
        k = w.floor()   # Integral part of w
        w = w - k       # Fractional part of w
        k = k.long()    # Transfer k to long int

        y = self.pyramid.repeat(num_area, 1, 1, 1, 1)    # A*N*T xLxCxHxW
        k = k.expand(n, 1, *y.shape[2:])    # A*N*T x1xCxHxW, channel dim: 1-->3

        y0 = torch.gather(y, 1, k)  # select low level, Nx1xCxHxW
        y1 = torch.gather(y, 1, torch.clamp(k + 1, max=self.num_levels - 1)) # select high level, Nx1xCxHxW

        # return ((1 - w) * y0 + w * y1).squeeze(dim=1)
        perturb_x = ((1 - w) * y0 + w * y1)    #Nx1xCxHxW 
        return perturb_x

    def to(self, dev):
        """Switch to another device.
        Args:
            dev: PyTorch device.
        Returns:
            Perturbation: self.
        """
        self.pyramid.to(dev)
        return self

    def __str__(self):
        return (
            f"Perturbation:\n"
            f"- type: {self.type}\n"
            f"- num_levels: {self.num_levels}\n"
            f"- pyramid shape: {list(self.pyramid.shape)}"
        )

class CoreLossCalulator:
    def __init__ (self, bs, nt, nrow, ncol, areas, device, 
                  num_key_frames=7, spatial_range=(11, 11),
                  core_shape='ellipsoid'):
        self.bs = bs
        self.nt = nt
        self.nrow = nrow
        self.ncol = ncol
        self.areas = areas
        self.narea = len(areas)
        self.device = device

        self.new_nrow = nrow
        self.new_ncol = ncol

        self.num_key_frames = num_key_frames
        self.spatial_range = spatial_range
        self.core_shape = core_shape
        self.conv_stride = spatial_range[0]
        self.core_kernel_ones = int(num_key_frames * spatial_range[0] * spatial_range[1] * 0.5236)
        self.core_kernels = []
        self.core_topks = []

        for a_idx, area in enumerate(areas):
            if self.core_shape == 'ellipsoid':
                core_kernel = self.get_ellipsoid_kernel([num_key_frames, spatial_range[0], spatial_range[1]]).to(device)
            elif self.core_shape == 'cylinder':
                core_kernel = self.get_cylinder_kernel([num_key_frames, spatial_range[0], spatial_range[1]]).to(device)
            else:
                raise Exception(f'Unsurported core_shape, given {self.core_shape}')
            self.core_kernels.append(core_kernel)
            self.core_topks.append(math.ceil((area*self.nt*self.new_nrow*self.new_ncol) / self.core_kernel_ones))

    # mask: AxNx1xTx S_out x S_out
    def calculate (self, masks):
        losses = []
        small_masks = masks

        losses = []
        for a_idx, area in enumerate(self.areas):
            core_kernel = self.core_kernels[a_idx]
            core_kernel_sum = core_kernel.sum()
            mask_conv = F.conv3d(small_masks[a_idx], core_kernel, bias=None, 
                                stride=(1, self.conv_stride, self.conv_stride), 
                                padding=0, dilation=1, groups=1)
            # mask_conv_sorted = mask_conv.view(self.bs, -1).sort(dim=1, descending=True)[0]
            mask_conv_sorted = mask_conv.view(self.bs, -1)
            shuffled_inds = torch.randperm(mask_conv_sorted.shape[1], device=mask_conv_sorted.device)
            mask_conv_sorted = mask_conv_sorted[:, shuffled_inds].sort(dim=1, descending=True)[0]
            # loss = ((mask_conv_sorted[:, :self.core_topks[a_idx]] - 1) ** 2).mean(dim=1) \
            #         + ((mask_conv_sorted[:, self.core_topks[a_idx]:] - 0) ** 2).mean(dim=1)
            loss = ((mask_conv_sorted[:, :self.core_topks[a_idx]]/core_kernel_sum - 1) ** 2).mean(dim=1) \
                    + ((mask_conv_sorted[:, self.core_topks[a_idx]:]/core_kernel_sum - 0) ** 2).mean(dim=1)
            losses.append(loss)
        losses = torch.stack(losses, dim=0)   # A x N
        return losses

    def get_ellipsoid_kernel(self, dims):
        assert(len(dims) == 3)
        d1, d2, d3 = dims
        v1, v2, v3 = np.meshgrid(np.linspace(-1, 1, d1), np.linspace(-1, 1, d2), np.linspace(-1, 1, d3), indexing='ij')
        dist = np.sqrt(v1 ** 2 + v2 ** 2 + v3 ** 2)
        return torch.from_numpy((dist <= 1)[np.newaxis, np.newaxis, ...]).float()

    def get_cylinder_kernel(self, dims):
        assert(len(dims) == 3)
        d1, d2, d3 = dims
        v1, v2, v3 = np.meshgrid(np.linspace(-1, 1, d1), np.linspace(-1, 1, d2), np.linspace(-1, 1, d3), indexing='ij')
        dist = np.sqrt(v2 ** 2 + v3 ** 2)
        return torch.from_numpy((dist <= 1)[np.newaxis, np.newaxis, ...]).float()

# Control smoothness
# masks: AxNx1xTx S_out x S_out
def diff_loss(masks, num_diff=3):
    narea = masks.shape[0]
    bs = masks.shape[1]
    diffs = [masks[:, :, :, 1:, ...] - masks[:, :, :, :-1, ...]]
    for _ in range(num_diff - 1):
        diffs.append(diffs[-1][:, :, :, 1:, ...] - diffs[-1][:, :, :, :-1, ...])
    diffs_mean = torch.stack([(item.view(narea, bs, -1) ** 2).mean(dim=2) for item in diffs], dim=0)
    return diffs_mean.mean(dim=0)   # A x N

# masks: AxNx1xTx S_out x S_out
# total_ones: A
def sum_loss(masks, total_ones):
    narea = masks.shape[0]
    bs = masks.shape[1]
    return (masks.view(narea, bs, -1).sum(dim=2) / total_ones.unsqueeze(-1) - 1) ** 2

def video_perturbation(model,
                          input,
                          target,
                          areas=[0.1],
                          perturb_type=BLUR_PERTURBATION,
                          max_iter=2000,
                          num_levels=8,
                          step=7,
                          sigma=11,
                          tsigma=0,
                          with_diff=False,
                          with_core=False,
                          variant=PRESERVE_VARIANT,
                          print_iter=None,
                          debug=False,
                          reward_func="simple_log",
                          resize=False,
                          resize_mode='bilinear',
                          smooth=0,
                          gpu_id=0,
                          core_num_keyframe=5,
                          core_spatial_size=11,
                          core_shape='ellipsoid'):
    
    if isinstance(areas, float):
        areas = [areas]
    momentum = 0.9
    learning_rate = 0.05
    regul_weight = 300
    reward_weight = 1
    core_weight = 0
    # core_weight = 300
    
    device = input.device
    torch.cuda.set_device(gpu_id)

    regul_weight_last = max(regul_weight / 2, 1)

    # input shape: NxCxTxHxW (1x3x16x112x112)
    batch_size = input.shape[0] # N
    num_frame = input.shape[2]  # T=16
    num_area = len(areas)       # A

    if debug:
        print(
            f"spatiotemporal_perturbation:\n"
            f"- target: {target}\n"
            f"- areas: {areas}\n"
            f"- variant: {variant}\n"
            f"- max_iter: {max_iter}\n"
            f"- step/sigma: {step}, {sigma}\n"
            f"- voxel size: {list(input.shape)}\n"
            f"- reward function: {reward_func}"
        )
    print(f"- Target: {target.detach().cpu().tolist()}")

    # Disable gradients for model parameters.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # y = model(input)
    # ymin, ymin_idx = torch.min(y, dim=1)
    # print(f'Min index: {ymin_idx[0].item()}, Min: {ymin[0].item()}')

    # NxCxTxHxW --> N*T x CxHxW
    pmt_inp = input.transpose(1,2).contiguous() # NxTxCxHxW
    pmt_inp = pmt_inp.view(batch_size*num_frame, *pmt_inp.shape[2:])  # N*T x CxHxW

    # Get the perturbation operator.
    # perturbation.pyramid: T*N x LxCxHxW
    perturbation = Perturbation(pmt_inp, num_levels=num_levels, 
                                    type=perturb_type).to(device)

    # Prepare the mask generator (generating mask(134x134) from pmask(16x16)).
    shape = perturbation.pyramid.shape[3:]  # 112x112
    mask_generator = MaskGenerator(shape, step, sigma, pooling_method='softmax').to(device)
    h, w = mask_generator.shape_in  # h=112/step, w=112/step, 16x16
    pmasks = torch.ones(num_area*batch_size*num_frame, 1, h, w).to(device)  #A*N*T x 1x16x16

    if with_core:
        mask_core = CoreLossCalulator(batch_size, num_frame, 
                            mask_generator.shape_out[0], mask_generator.shape_out[1], 
                            areas, device, num_key_frames=core_num_keyframe,
                            spatial_range=(core_spatial_size, core_spatial_size),
                            core_shape=core_shape)

    max_area = np.prod(mask_generator.shape_out)
    max_volume = np.prod(mask_generator.shape_out) * num_frame
    total_ones = torch.zeros(num_area).to(device)
    # Prepare reference area vector.
    vref = torch.ones(num_area, batch_size, max_volume).to(device)
    for a_idx, area in enumerate(areas):
        total_ones[a_idx] = int(area * max_volume)
        vref[a_idx, :, :int(max_volume * (1 - area))] = 0

    # Initialize optimizer.
    optimizer = optim.SGD([pmasks],
                          lr=learning_rate,
                          momentum=momentum,
                          dampening=momentum)
    hist = torch.zeros((num_area, batch_size, 2, 0))

    sum_time = 0
    for t in range(max_iter):
        end_time = time.time()

        pmasks.requires_grad_(True)
        masks, padded_masks = mask_generator.generate(pmasks)

        if variant == DELETE_VARIANT:
            perturb_x = perturbation.apply(1 - masks)  # A*N*T x 1xCxHxW
        elif variant == PRESERVE_VARIANT:
            perturb_x = perturbation.apply(masks)  # A*N*T x 1xCxHxW
        elif variant == DUAL_VARIANT:
            perturb_x = torch.cat((
                perturbation.apply(masks),  #preserve
                perturbation.apply(1 - masks),  #delete
            ), dim = 1) # A*N*T x 2xCxHxW
        else:
            assert False

        perturb_x = perturb_x.view(num_area, batch_size, num_frame, *perturb_x.shape[1:]) # AxNxTx2xCxHxW
        perturb_x = perturb_x.permute(3,0,1,4,2,5,6).contiguous()   # 2xAxNxCxTxHxW
        perturb_x = perturb_x.view(perturb_x.shape[0]*num_area*batch_size, *perturb_x.shape[3:])    # 2*A*N x CxTxHxW

        masks = masks.view(num_area, batch_size, num_frame, *masks.shape[1:]).transpose(2,3) # AxNx1xTxHxW
        padded_masks = padded_masks.view(num_area, batch_size, num_frame, \
                                *padded_masks.shape[1:]).transpose(2,3)    # AxNx1xTx S_out x S_out

        # Evaluate the model on the masked data
        # The input of model should have size of NxCxTxHxW
        y = model(perturb_x)    # 2*A*N x num_classes, default the model has softmax

        # Cal probability
        prob, pred_label, pred_label_prob = process_activations(y, target, softmaxed=True)

        # Get reward.
        if reward_func == "simple":
            reward = simple_reward(y, target, variant=variant)
        elif reward_func == "contrastive":
            reward = contrastive_reward(y, target, variant=variant)
        elif reward_func == "simple_log":
            reward = simple_log_reward(y, target, variant=variant)  # 2*A*N
        reward = reward.view(-1, num_area, batch_size).mean(dim=0) * reward_weight  # A x N

        # Area regularization.
        # padded_masks: A x N x 1 x T x S_out x S_out
        # mask_sorted = padded_masks.squeeze(2).reshape(num_area, batch_size, -1).sort(dim=2)[0]  # A x N x T*S_out*S_out
        mask_sorted = padded_masks.squeeze(2).reshape(num_area, batch_size, -1)  # A x N x T*S_out*S_out
        shuffled_inds = torch.randperm(mask_sorted.shape[2], device=mask_sorted.device)
        mask_sorted = mask_sorted[:,:,shuffled_inds]
        mask_sorted = mask_sorted.sort(dim=2)[0]

        regul = ((mask_sorted - vref)**2).mean(dim=2) * regul_weight # A x N
        if with_diff:
            regul += diff_loss(padded_masks) * regul_weight
        if with_core:
            regul += mask_core.calculate(padded_masks.contiguous()) * core_weight

        # Energy summary
        energy = (reward + regul).sum() 

        # Record energy
        # hist: num_area x batch_size x 2 x num_iter
        hist_item = torch.cat((reward.detach().cpu().view(num_area, batch_size, 1, 1),
                            regul.detach().cpu().view(num_area, batch_size, 1, 1)), dim=2)
        hist = torch.cat((hist, hist_item), dim=3)

        # Gradient step.
        optimizer.zero_grad()
        energy.backward()
        optimizer.step()

        pmasks.data = pmasks.data.clamp(0, 1)

        regul_weight *= 1.0008668     #1.00069^800=2
        # regul_weight *= 1.001734
        if t == 100:
            core_weight = 300

        sum_time += time.time() - end_time
        # Print iteration information
        if (print_iter != None) and (t % print_iter == 0):
            print(f"[{t+1:04d}/{max_iter:04d}]", end="\n")
            for i in range(batch_size):
                for a_idx, area in enumerate(areas):
                    if variant == "dual":
                        print(f" [GPU: {gpu_id} area:{area:.2f} loss:{hist[a_idx,i,0,-1]:.2f} reg:{hist[a_idx,i,1,-1]:.2f} "\
                            f"presv:{prob[a_idx*batch_size+i]:.2f}/{pred_label[a_idx*batch_size+i]} "\
                            f"del:{prob[(num_area+a_idx)*batch_size+i]:.2f}/{pred_label[(num_area+a_idx)*batch_size+i]}]", end="")
                    else:
                        print(f" [GPU: {gpu_id} area:{area:.2f} loss:{hist[a_idx,i,0,-1]:.2f} reg:{hist[a_idx,i,1,-1]:.2f} "\
                            f"{variant}:{prob[a_idx*batch_size+i]:.2f}/{pred_label[a_idx*batch_size+i]}]", end="")
                    print()
            avg_time = sum_time/(t+1)
            print(f"GPU: {gpu_id}, Average time: {avg_time:.3f}.")
            
    masks = masks.detach()
    # Resize saliency map.
    list_mask = []
    for a_idx, area in enumerate(areas):
        area_mask = []
        for frame_idx in range(num_frame):
            mask = masks[a_idx,:,:,frame_idx,:,:]   # NxCxHxW
            mask = resize_saliency(pmt_inp, mask, resize, mode=resize_mode)
            # Smooth saliency map.
            if smooth > 0:
                mask = imsmooth(mask, sigma=smooth * min(mask.shape[2:]), padding_mode='constant')
            area_mask.append(mask)
        area_mask = torch.stack(area_mask, dim=2)   # NxCxTxHxW
        list_mask.append(area_mask)
    masks = torch.stack(list_mask, dim=1).cpu()   # NxAxCxTxHxW

    # masks: AxNxCxTxHxW; hist: AxNx2xmax_iter; perturb_x: 2*A*N x CxTxHxW
    return masks, hist