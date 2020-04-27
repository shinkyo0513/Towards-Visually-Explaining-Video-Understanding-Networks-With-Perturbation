import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# output: batch_size x num_class(1000)
# target: batch_size x 1 (LongTensor)


def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def process_activations(activations, targets, softmaxed=True):
    assert activations.shape[0] % targets.shape[0] == 0, \
        f"Check the batch size of activations and targets!"

    b_repeat = activations.shape[0] // targets.shape[0]
    if b_repeat > 1:
        targets = targets.repeat(b_repeat)

    if not softmaxed:
        soft_act = torch.nn.functional.softmax(activations, dim=1)
    else:
        soft_act = activations

    row_idx = torch.arange(
        soft_act.shape[0], dtype=torch.long, device=activations.device)
    probs = soft_act[row_idx, targets]  # batch_size
    pred_label_probs, pred_labels = torch.max(soft_act, dim=1)

    return probs, pred_labels, pred_label_probs

if __name__ == "__main__":
    output = torch.tensor([[0.01, 0.1, 0.05, 0.9], [0.99, 0.4, 0.2, 0.3]])
    target = torch.LongTensor([[3], [1]])

    acc = accuracy(output, target, topk=(1, 3))
    print(acc)
