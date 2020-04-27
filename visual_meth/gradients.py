import torch
import numpy as np 

def gradients (inputs, labels, model, device, multiply_input=False,
                 polarity='both', show_gray=False):
    model.eval()   # Set model to evaluate mode

    bs, ch, nt, h, w = inputs.shape
    assert ch == 3
    assert labels.shape[0] == bs

    inputs = inputs.to(device)
    # labels = labels.to(device)
    labels = labels.to(dtype=torch.long)

    # Get model outputs and calculate loss
    inputs.requires_grad = True
    with torch.set_grad_enabled(True):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        backward_signals = torch.zeros_like(outputs, device=device)
        for bidx in range(bs):
            backward_signals[bidx, labels[bidx].cpu().item()] = 1.0
        outputs.backward(backward_signals)
    inputs_grad = inputs.grad.cpu()
    if multiply_input:
        inputs_grad *= inputs.detach().cpu()

    if polarity == 'positive':
        inputs_grad = torch.clamp(inputs_grad, min=0.0)
    elif polarity == 'negative':
        inputs_grad = -1.0 * torch.clamp(inputs_grad, max=0.0)

    if show_gray:
        inputs_grad = inputs_grad.mean(dim=1, keepdim=True).repeat(1,ch,1,1,1)

    grad_sorted = inputs_grad.view(-1).sort()[0]
    length = grad_sorted.shape[0]
    grad_min = 0.0
    grad_max = grad_sorted[int(0.999 * length)]
    grad_show = inputs_grad.clamp(min=grad_min, max=grad_max)
    grad_show = (grad_show - grad_min) / (grad_max - grad_min)
    # print(grad_show.max(), grad_show.min())

    if polarity == 'positive':
        grad_show[:,1:,...] = 0.0
    elif polarity == 'negative':
        grad_show[:,:2,...] = 0.0

    return grad_show
