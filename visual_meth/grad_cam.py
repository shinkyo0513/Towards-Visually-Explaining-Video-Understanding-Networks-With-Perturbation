import torch
import torch.nn.functional as F
import numpy as np 

# Backward hook
observ_grad_ = []
def backward_hook(m, i_grad, o_grad): 
    global observ_grad_
    observ_grad_.insert(0, o_grad[0].detach())

# Forward hook
observ_actv_ = []
def forward_hook(m, i, o):
    global observ_actv_
    observ_actv_.append(o.detach())

def grad_cam (inputs, labels, model, device, layer_name, norm_vis=True):
    model.eval()   # Set model to evaluate mode
    
    bs, ch, nt, h, w = inputs.shape
    assert ch == 3
    assert labels.shape[0] == bs

    # layer_dict = dict(model.module.named_children())
    # assert layer_name in layer_dict, \
    #     f'Given layer ({layer_name}) is not in model. {model}'
    # observ_layer = layer_dict[layer_name]

    observ_layer = model
    for name in layer_name:
        # print(dict(observ_layer.named_children()).keys())
        observ_layer = dict(observ_layer.named_children())[name]

    observ_layer.register_backward_hook(backward_hook)
    observ_layer.register_forward_hook(forward_hook)

    inputs = inputs.to(device)
    labels = labels.to(dtype=torch.long)

    # Forward pass
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    observ_actv = observ_actv_[0]   # 1 x C x num_f/8 x 56 x 56
    # print('observ_actv:', observ_actv.shape)
    observ_actv = torch.repeat_interleave(observ_actv, int(nt/observ_actv.shape[2]), dim=2)

    # backward pass
    backward_signals = torch.zeros_like(outputs, device=device)
    for bidx in range(bs):
        backward_signals[bidx, labels[bidx].cpu().item()] = 1.0
    outputs.backward(backward_signals)

    observ_grad = observ_grad_[0]   # 1 x C x num_f/8 x 56 x 56
    # print('observ_grad:', observ_grad.shape)
    observ_grad = torch.repeat_interleave(observ_grad, int(nt/observ_grad.shape[2]), dim=2)

    observ_grad_w = observ_grad.mean(dim=4, keepdim=True).mean(dim=3, keepdim=True) # 1 x 512 x num_f x 1x1
    out_masks = F.relu( (observ_grad_w*observ_actv).sum(dim=1, keepdim=True) ) # 1 x 1 x num_f x 14x14

    if norm_vis:
        out_masks = (out_masks - out_masks.min()) / (out_masks.max() - out_masks.min())

    return out_masks



