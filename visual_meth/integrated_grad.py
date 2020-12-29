import torch
import numpy as np 

def integrated_grad_show (grads, polarity):
    positive_channel = [0, 1, 0]  # Green
    negative_channel = [1, 0, 0]  # Red
    assert polarity in ['both', 'positive', 'negative']

    if polarity == 'both':
        grads = torch.abs(grads)
        channel = torch.tensor([1,1,1]).reshape(1,3,1,1,1)
    elif polarity == 'positive':
        grads = torch.clamp(grads, min=0.0)
        channel = torch.tensor(positive_channel).reshape(1,3,1,1,1)
    elif polarity == 'negative':
        grads = -1.0 * torch.clamp(grads, max=0.0)
        channel = torch.tensor(negative_channel).reshape(1,3,1,1,1)
    grads = grads.mean(dim=1, keepdim=True)  # convert to gray, Nx1xTxHxW

    grads_np = grads.cpu().numpy()
    vmax = np.percentile(grads_np, 99.9)
    vmin = np.min(grads_np)
    grad_show_np = np.clip((grads_np - vmin) / (vmax - vmin), 0, 1)    # Nx1xTxHxW
    grad_show = torch.from_numpy(grad_show_np.repeat(3, axis=1)) * channel    # Nx3xTxHxW
    return grad_show

def integrated_grad (inputs, labels, model, device, steps, 
                        polarity='both', show_gray=True):
    model.eval()   # Set model to evaluate mode

    bs, ch, nt, h, w = inputs.shape
    assert ch == 3
    assert labels.shape[0] == bs

    inputs = inputs.to(device)
    # labels = labels.to(device)
    labels = labels.to(dtype=torch.long)

    baseline = torch.zeros_like(inputs, device=device)
    baseline[:, 0, ...] = -1.8952
    baseline[:, 1, ...] = -1.7822
    baseline[:, 2, ...] = -1.7349
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
        
    # if polarity == 'positive':
    #     intg_grads = torch.clamp(intg_grads, min=0.0)
    # elif polarity == 'negative':
    #     intg_grads = -1.0 * torch.clamp(intg_grads, max=0.0)

    # if show_gray:
    #     intg_grads = intg_grads.mean(dim=1, keepdim=True).repeat(1,ch,1,1,1)

    # grad_sorted = intg_grads.view(-1).sort()[0]
    # length = grad_sorted.shape[0]
    # grad_min = 0.0
    # grad_max = grad_sorted[int(0.999 * length)]
    # grad_show = intg_grads.clamp(min=grad_min, max=grad_max)
    # grad_show = (grad_show - grad_min) / (grad_max - grad_min)  # NxCxTxHxW
    
    # if polarity == 'positive':
    #     intg_grads[:,1:,...] = 0.0
    # elif polarity == 'negative':
    #     intg_grads[:,:2,...] = 0.0

    grad_show = integrated_grad_show(intg_grads, polarity)
    return grad_show

