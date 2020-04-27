import math
import matplotlib.pyplot as plt
import numpy as np

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
    def __init__(self, shape, step, sigma, batch_size=1, clamp=True, pooling_method='softmax'):
        self.shape = shape
        self.step = step
        self.sigma = sigma
        self.coldness = 20
        self.batch_size = batch_size
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
        # mask: Nx1xHxW
        n = mask.shape[0]
        # starred expression: unpack a list to separated numbers
        w = mask.reshape(n, 1, *mask.shape[1:]) # Nx1x1xHxW, mask.unsqueeze(1)
        w = w * (self.num_levels - 1)   # w = 7*w
        k = w.floor()   # Integral part of w
        w = w - k       # Fractional part of w
        k = k.long()    # Transfer k to long int

        # y = self.pyramid[None, :] #1xLxCxHxW
        # y = y.expand(n, *y.shape[1:]) #nxLxCxHxW
        # k = k.expand(n, 1, *y.shape[2:])  #nx1xCxHxW
        y = self.pyramid    # NxLxCxHxW
        k = k.expand(n, 1, *y.shape[2:])    #Nx1xCxHxW
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

def video_perturbation(model, input, target,
                        areas=[0.1], method='spatiotemporal', perturb_type=FADE_PERTURBATION,
                        max_iter=2000, num_levels=8, step=7, sigma=11, variant=PRESERVE_VARIANT,
                        print_iter=None, debug=False, reward_func="simple_log", resize=False,
                        resize_mode='bilinear', smooth=0, num_devices=1):
    
    if isinstance(areas, float):
        areas = [areas]
    momentum = 0.9
    learning_rate = 0.05
    regul_weight = 300
    reward_weight = 100
    device = input.device

    iter_period = 2000

    regul_weight_last = max(regul_weight / 2, 1)

    # input shape: NxCxTxHxW (1x3x16x112x112)
    batch_size = input.shape[0]
    num_frame = input.shape[2]  #16
    num_areas = len(areas)

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

    ori_y = model(input)
    ori_prob, ori_pred_label, _ = process_activations(ori_y, target, softmaxed=True)

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
    pmasks = torch.ones(batch_size*num_frame, 1, h, w).to(device)  #N*T x 1x16x16

    max_area = np.prod(mask_generator.shape_out)
    max_volume = np.prod(mask_generator.shape_out) * num_frame

    # Prepare reference area vector.
    if method == 'spatiotemporal':
        vref = torch.ones(batch_size, max_volume).to(device)
        vref[:, :int(max_volume * (1 - areas[0]))] = 0
    elif method == 'extremal':
        aref = torch.ones(batch_size, num_frame, max_area).to(device)
        aref[:, :, :int(max_area * (1 - areas[0]))] = 0

    # Initialize optimizer.
    optimizer = optim.SGD([pmasks],
                          lr=learning_rate,
                          momentum=momentum,
                          dampening=momentum)
    hist = torch.zeros((batch_size, 2, 0))

    for t in range(max_iter):
        pmasks.requires_grad_(True)
        masks, padded_masks = mask_generator.generate(pmasks)

        if variant == DELETE_VARIANT:
            perturb_x = perturbation.apply(1 - masks)  # N*T x 1xCxHxW
        elif variant == PRESERVE_VARIANT:
            perturb_x = perturbation.apply(masks)  # N*T x 1xCxHxW
        elif variant == DUAL_VARIANT:
            perturb_x = torch.cat((
                perturbation.apply(masks),  #preserve
                perturbation.apply(1 - masks),  #delete
            ), dim = 1) # N*T x 2xCxHxW
        else:
            assert False

        perturb_x = perturb_x.view(batch_size, num_frame, *perturb_x.shape[1:]) # NxTx2xCxHxW
        perturb_x = perturb_x.permute(2, 0, 3, 1, 4, 5).contiguous()   # 2xNxCxTxHxW
        perturb_x = perturb_x.view(perturb_x.shape[0]*perturb_x.shape[1], *perturb_x.shape[2:])    # 2*N x CxTxHxW

        masks = masks.view(batch_size, num_frame, *masks.shape[1:]).transpose(1,2) # Nx1xTxHxW
        padded_masks = padded_masks.view(batch_size, num_frame, \
                                *padded_masks.shape[1:]).transpose(1,2)    # Nx1xTx S_out x S_out

        # Evaluate the model on the masked data
        # The input of model should have size of NxCxTxHxW
        y = model(perturb_x)    # 2*N x num_classes
        y = F.softmax(y, dim=1)

        # Cal probability
        prob, pred_label, pred_label_prob = process_activations(y, target, softmaxed=True)

        # Get reward.
        if reward_func == "simple_log":
            reward = simple_log_reward(y, target, variant=variant)
        else:
            raise Exception("Do not support other reward function now.")
        # print(f"Reward shape: {reward.shape}")
        reward = reward.view(batch_size, -1).mean(dim=1) * reward_weight  # batch_size

        # Area regularization.
        # padded_masks: N x 1 x T x S_out x S_out
        if method == 'spatiotemporal':
            mask_sorted = padded_masks.squeeze(1).reshape(batch_size, -1).sort(dim=1)[0]
            regul = ((mask_sorted - vref)**2).mean(dim=1) * regul_weight # batch_size
        elif method == 'extremal':
            mask_sorted = padded_masks.squeeze(1).reshape(batch_size, num_frame, -1).sort(dim=2)[0]
            regul = ((mask_sorted - aref)**2).mean(dim=2).mean(dim=1) * regul_weight # batch_size

        # Energy summary
        energy = (reward + regul).sum() 

        # Gradient step.
        optimizer.zero_grad()
        energy.backward()
        optimizer.step()

        pmasks.data = pmasks.data.clamp(0, 1)

        # Record energy
        # hist: batch_size x 2 x num_iter
        hist_item = torch.cat(
                    (
                        reward.detach().cpu().view(-1, 1, 1),
                        regul.detach().cpu().view(-1, 1, 1)
                    ), dim=1)
        # print(f"Item shape: {hist_item.shape}")
        hist = torch.cat((hist,hist_item), dim=2)

        # Diagnostics.
        # debug_this_iter = debug and (regul_weight / regul_weight_last >= 2)

        # if (print_iter is not None and t % print_iter == 0) or debug_this_iter:
        if (print_iter != None) and (t % print_iter == 0):
            print("[{:04d}/{:04d}]".format(t + 1, max_iter), end="\n")
            for i in range(batch_size):
                if variant == "dual":
                    print(" [area:{:.2f} loss:{:.2f} reg:{:.2f} presv:{:.2f}/{} del:{:.2f}/{}]".format(
                        areas[0], hist[i, 0, -1], hist[i, 1, -1], 
                        prob[i], pred_label[i], 
                        prob[i+batch_size], pred_label[i+batch_size]), end="")
                else:
                    print(" [area:{:.2f} loss:{:.2f} reg:{:.2f} {}:{:.2f}/{}]".format(
                        areas[0], hist[i, 0, -1], hist[i, 1, -1],
                        variant, prob[i], pred_label[i]), end="")
                print()

        # if debug_this_iter:
        #     regul_weight_last = regul_weight
        #     # for i, a in enumerate(areas):
        #     for i in range(batch_size):
        #         plt.figure(i, figsize=(20, 6))
        #         plt.clf()
        #         ncols = 4 if variant == DUAL_VARIANT else 3
        #         plt.subplot(1, ncols, 1)
        #         plt.plot(hist[i, 0].numpy())
        #         plt.plot(hist[i, 1].numpy())
        #         plt.plot(hist[i].sum(dim=0).numpy())
        #         plt.legend(('energy', 'regul', 'both'))
        #         plt.title(f'target area:{areas[0]:.2f}')
        #         mask = padded_masks[i,:,7,:,:]
        #         plt.subplot(1, ncols, 2)
        #         imsc(mask, lim=[0, 1])
        #         plt.title(
        #             f"min:{mask.min().item():.2f}"
        #             f" max:{mask.max().item():.2f}"
        #             f" area:{mask.sum() / mask.numel():.2f}")
        #         plt.subplot(1, ncols, 3)
        #         imsc(perturb_x[i,:,7,:,:])
        #         if variant == DUAL_VARIANT:
        #             plt.subplot(1, ncols, 4)
        #             imsc(perturb_x[i + batch_size,:,7,:,:])
        #         plt.pause(0.001)

    masks = masks.detach()
    # Resize saliency map.
    list_mask = []
    for frame_idx in range(num_frame):
        mask = masks[:,:,frame_idx,:,:]
        mask = resize_saliency(pmt_inp, mask, resize, mode=resize_mode)
        # Smooth saliency map.
        if smooth > 0:
            mask = imsmooth(mask, sigma=smooth * min(mask.shape[2:]), padding_mode='constant')
        list_mask.append(mask)
    masks = torch.stack(list_mask, dim=2)   # NxCxTxHxW

    # masks: NxCxTxHxW; hist: Nx3xmax_iter
    return masks, hist, perturb_x