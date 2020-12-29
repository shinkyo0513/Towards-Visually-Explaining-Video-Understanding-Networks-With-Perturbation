import torch
import numpy as np 

def smooth_grad (inputs, labels, model, device, 
                    nsamples=25, variant=None, stdev_spread=0.15):
    model.eval()   # Set model to evaluate mode

    bs, ch, nt, h, w = inputs.shape
    assert ch == 3
    assert labels.shape[0] == bs
    assert variant in [None, 'square', 'variance']

    inputs = inputs.to(device)
    # labels = labels.to(device)
    labels = labels.to(dtype=torch.long)

    inputs_max = torch.max(inputs.view(bs, -1), dim=1)[0]
    inputs_min = torch.min(inputs.view(bs, -1), dim=1)[0]
    stdev = stdev_spread * (inputs_max - inputs_min)    # bs
    stdev = stdev.view(-1, 1).expand(-1, ch*nt*h*w)

    outputs = model(inputs)
    backward_signals = torch.zeros_like(outputs, device=device)

    for bidx in range(bs):
        backward_signals[bidx, labels[bidx].cpu().item()] = 1.0

    all_grads = []
    for i in range(nsamples):
        noise = torch.normal(0.0, stdev).to(device).reshape(bs, ch, nt, h, w)
        # print(noise.shape, noise.min(), noise.max())
        noisy_inputs = inputs + noise
        noisy_inputs.requires_grad_()

        # Forward
        outputs = model(noisy_inputs)
        _, preds = torch.max(outputs, dim=1)

        # Backward
        outputs.backward(backward_signals)
        noisy_grads = noisy_inputs.grad.cpu()
        all_grads.append(noisy_grads)
    all_grads = torch.stack(all_grads, dim=1)   # bs x nsamples x ch x nt x h x w
    
    if variant == None:
        smth_grad = torch.mean(all_grads, dim=1)
    elif variant == 'square':
        smth_grad = torch.mean(all_grads ** 2, dim=1)
    elif variant == 'variance':
        smth_grad = torch.var(all_grads, dim=1)
    smth_grad = smth_grad.numpy()   # bs x ch x nt x h x w

    grad_show = np.sum(np.abs(smth_grad), axis=1, keepdims=True)
    vmax = np.percentile(grad_show, 99.9)
    vmin = np.min(grad_show)
    grad_show = np.clip((grad_show - vmin) / (vmax - vmin), 0, 1)   # Nx1xTxHxW
    grad_show = torch.from_numpy(np.repeat(grad_show, 3, axis=1))
    return grad_show

