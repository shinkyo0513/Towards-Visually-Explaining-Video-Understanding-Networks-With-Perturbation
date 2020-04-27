import torch
import numpy as np 

def integrated_grad (inputs, labels, model, device, steps, 
                        polarity='positive', show_gray=True):
    model.eval()   # Set model to evaluate mode

    bs, ch, nt, h, w = inputs.shape
    assert ch == 3
    assert labels.shape[0] == bs

    inputs = inputs.to(device)
    # labels = labels.to(device)
    labels = labels.to(dtype=torch.long)

    baseline = torch.zeros_like(inputs, device=device)
    baseline_out = model(baseline)

    backward_signals = torch.zeros_like(baseline_out, device=device)
    for bidx in range(bs):
        backward_signals[bidx, labels[bidx].cpu().item()] = 1.0

    intg_grads = 0
    for i in range(steps):
        scaled_inputs = baseline + (float(i) / steps) * (inputs - baseline)
        scaled_inputs.requires_grad = True

        # Forward
        outputs = model(scaled_inputs)
        _, preds = torch.max(outputs, dim=1)

        # Backward
        outputs.backward(backward_signals)
        intg_grads += scaled_inputs.grad.cpu() / steps
    intg_grads *= (inputs - baseline).detach().cpu()
        
    if polarity == 'positive':
        intg_grads = torch.clamp(intg_grads, min=0.0)
    elif polarity == 'negative':
        intg_grads = -1.0 * torch.clamp(intg_grads, max=0.0)

    if show_gray:
        intg_grads = intg_grads.mean(dim=1, keepdim=True).repeat(1,ch,1,1,1)

    grad_sorted = intg_grads.view(-1).sort()[0]
    length = grad_sorted.shape[0]
    grad_min = 0.0
    grad_max = grad_sorted[int(0.999 * length)]
    grad_show = intg_grads.clamp(min=grad_min, max=grad_max)
    grad_show = (grad_show - grad_min) / (grad_max - grad_min)  # NxCxTxHxW
    
    if polarity == 'positive':
        intg_grads[:,1:,...] = 0.0
    elif polarity == 'negative':
        intg_grads[:,:2,...] = 0.0
    return grad_show

