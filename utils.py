import sys
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from typing import Any
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader


class Normalize(nn.Module):

    def __init__(self, mean=(0.4850, 0.4560, 0.4060), std=(0.2290, 0.2240, 0.2250)):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


def get_accuracy(model, x_orig, y_orig, bs=64, device=torch.device('cuda:0')):
    n_batches = x_orig.shape[0] // bs
    acc = 0.
    wrong_pred = []
    for counter in range(n_batches):
        x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        output = model(x)
        acc += (output.max(1)[1] == y).float().sum()
        wrong_pred.append(~(output.max(1)[1] == y))
    
    return (acc / x_orig.shape[0]).item(), torch.cat(wrong_pred, dim=0)

def get_pred(model, x):
    return model(x).max(1)[1]

def get_transforms(domain='imval'):
    if domain == 'imval':
        return transforms.Compose([
            transforms.Resize(256),  # resize shorter
            transforms.CenterCrop(224),  # take center crop
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
        ])

    else:
        raise NotImplementedError

def load_data(domain='imagenet', batch_size=128):
    if domain == 'imagenet':
        val_transforms = get_transforms(domain='imval')
        val_data = ImageFolder(root='', 
                            transform=val_transforms,)
        sub_val_dataset = Subset(val_data, torch.randint(low=0, high=5000, size=(1000,)))
        val_loader = DataLoader(sub_val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
        return val_loader
    else:
        raise NotImplementedError
    

class Logger(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()