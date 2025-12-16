import sys
import torch
import torch.nn as nn
import timm
import random
import torchvision.transforms as transforms
from typing import Any
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100
from torch.utils.data import Subset, DataLoader
from PIL import ImageFilter
from tqdm import tqdm
import os
import torchvision.transforms.functional as tf

from tvm_transform import Defense, defense

result_dir = './purification_result/rest50/'

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

def run_purifier(purifier, advs, eps, arch, bs=64, device=torch.device('cuda:0')):
    n_batches = advs.shape[0] // bs
    batch_fine, batch_coarse = [], []
    for counter in range(n_batches):
        adv = advs[counter * bs:min((counter + 1) * bs, advs.shape[0])].clone().to(device)
        fine, coarse = purifier(adv, eps=eps, arch=arch)
        batch_fine.append(fine)
        batch_coarse.append(coarse)
    return torch.cat(batch_fine, dim=0), torch.cat(batch_coarse, dim=0)

def run_coarse_purifier(purifier, advs, eps, arch, bs=64, device=torch.device('cuda:0')):
    n_batches = advs.shape[0] // bs
    batch_fine, batch_coarse = [], []
    for counter in range(n_batches):
        adv = advs[counter * bs:min((counter + 1) * bs, advs.shape[0])].clone().to(device)
        _, coarse = purifier(adv, eps=eps, arch=arch)
        batch_coarse.append(coarse)
    return torch.cat(batch_coarse, dim=0)

def debug_purifier(purifier, advs, nat_x, eps, arch, bs=64, device=torch.device('cuda:0')):
    n_batches = advs.shape[0] // bs
    batch_fine, batch_coarse = [], []
    for counter in range(n_batches):
        adv = advs[counter * bs:min((counter + 1) * bs, advs.shape[0])].clone().to(device)
        nat = nat_x[counter * bs:min((counter + 1) * bs, advs.shape[0])].clone().to(device)
        fine, coarse = purifier(adv, eps=eps, arch=arch, nat_x=nat)
        batch_fine.append(fine)
        batch_coarse.append(coarse)
    return torch.cat(batch_fine, dim=0), torch.cat(batch_coarse, dim=0)

def run_fine(purifier, advs, coarse, eps, arch, bs=64, device=torch.device('cuda:0')):
    n_batches = advs.shape[0] // bs
    batch_fine = []
    for counter in range(n_batches):
        adv = advs[counter * bs:min((counter + 1) * bs, advs.shape[0])].clone().to(device)
        fine = purifier(adv, coarse=coarse, eps=eps, arch=arch)
        batch_fine.append(fine)
    return torch.cat(batch_fine, dim=0)

def run_apgd(adversary, purified_model, x_val, y_val, eps, n_iter, bs=64, device=torch.device('cuda:0')):
    n_batches = x_val.shape[0] // bs
    batch_advs = []
    for counter in tqdm(range(n_batches)):
        x = x_val[counter * bs:min((counter + 1) * bs, x_val.shape[0])].clone().to(device)
        y = y_val[counter * bs:min((counter + 1) * bs, x_val.shape[0])].clone().to(device)
        _, _, _, advs = adversary(purified_model, x, y, 'Linf', eps, n_iter=n_iter)
        batch_advs.append(advs)
    return torch.cat(batch_advs, dim=0)


def run_difgsm(adversary, x_val, y_val, bs=64, device=torch.device('cuda:0')):
    n_batches = x_val.shape[0] // bs
    batch_advs = []
    for counter in tqdm(range(n_batches)):
        x = x_val[counter * bs:min((counter + 1) * bs, x_val.shape[0])].clone().to(device)
        y = y_val[counter * bs:min((counter + 1) * bs, x_val.shape[0])].clone().to(device)
        advs = adversary(x, y)
        batch_advs.append(advs)
    return torch.cat(batch_advs, dim=0)

def run_pgd(model, adversary, criterion, x_val, y_val, bs=64, device=torch.device('cuda:0')):
    n_batches = x_val.shape[0] // bs
    batch_advs = []
    for counter in tqdm(range(n_batches)):
        x = x_val[counter * bs:min((counter + 1) * bs, x_val.shape[0])].clone().to(device)
        y = y_val[counter * bs:min((counter + 1) * bs, x_val.shape[0])].clone().to(device)
        advs = adversary(model, criterion, x, y) + x
        batch_advs.append(advs)
    return torch.cat(batch_advs, dim=0)

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
    elif domain == 'cifar':
        return transforms.Compose([transforms.ToTensor()])

    else:
        raise NotImplementedError

def load_data(domain='imagenet'):
    if domain == 'imagenet':
        val_transforms = get_transforms(domain='imval')
        val_data = ImageFolder(root='', 
                            transform=val_transforms,)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False,
                                num_workers=0, pin_memory=True)
        return val_loader, val_data.classes
    elif domain == 'cifar10':
        val_transforms = get_transforms(domain='cifar')
        val_data = CIFAR10('', train=False, 
                           transform=val_transforms, download=True)
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False,
                               pin_memory=True, num_workers=4)
        return val_loader
    elif domain == 'sub-cifar10':
        val_transforms = get_transforms(domain='cifar')
        val_data = CIFAR10('', train=False, 
                           transform=val_transforms, download=True)
        sub_val_dataset = Subset(val_data, torch.randint(low=0, high=10000, size=(512,)))
        val_loader = DataLoader(sub_val_dataset, batch_size=len(sub_val_dataset), shuffle=False,
                                num_workers=4, pin_memory=True)
        return val_loader
    elif domain == 'sub-cifar100':
        val_transforms = get_transforms(domain='cifar')
        val_data = CIFAR100('', train=False, 
                           transform=val_transforms, download=True)
        sub_val_dataset = Subset(val_data, torch.randint(low=0, high=10000, size=(512,)))
        val_loader = DataLoader(sub_val_dataset, batch_size=len(sub_val_dataset), shuffle=False,
                                num_workers=4, pin_memory=True)
        return val_loader
    elif domain == 'cifar100':
        val_transforms = get_transforms(domain='cifar')
        val_data = CIFAR100('', train=False, 
                           transform=val_transforms, download=True)
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False,
                               pin_memory=True, num_workers=4)
        return val_loader
    else:
        raise NotImplementedError
    
defense = Defense(defense, 'tvm')
class WeakTransform():
    def __init__(self, opt='gaussian'):
        super().__init__()

        if opt == 'gaussian':
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomApply([_MedianFilter(size=3)], p=1),
                transforms.RandomApply([GaussianBlur(sigma=[1.8, 1.8])], p=1),
                transforms.ToTensor(),
                # defense,
            ])
        elif opt == 'median':
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.RandomApply([_MedianFilter(size=3)], p=1),
                transforms.RandomApply([GaussianBlur(sigma=[1.8, 1.8])], p=1),
                transforms.ToTensor(),
                # defense,
            ])
        else:
            raise NotImplementedError

    def __call__(self, x_tensor):
        device = x_tensor.device
        x_tensor = x_tensor.cpu()
        x = torch.unbind(x_tensor, dim=0)
        x = [self.transforms(img).unsqueeze(0) for img in x]
        x = torch.cat(x, dim=0)

        return x.float().to(device)


class ILAProjLoss(torch.nn.Module):
    def __init__(self):
        super(ILAProjLoss, self).__init__()
    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        n = old_attack_mid.shape[0]
        x = (old_attack_mid - original_mid).reshape(n, -1) 
        y = (new_mid - original_mid).reshape(n, -1)        
        proj_loss = torch.sum(y * x) / n
        return proj_loss
    
class NormalizedILAProjLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, old_attack_mid, new_mid, original_mid, eps=1e-1):
        n = old_attack_mid.shape[0]
        x = (old_attack_mid - original_mid).reshape(n, -1)
        x_factor = torch.sqrt(torch.sum(x ** 2)) + eps
        normalized_x = x / x_factor

        if new_mid.equal(original_mid):
            normalized_y = (new_mid - original_mid).reshape(n, -1)
        else:
            y = (new_mid - original_mid).reshape(n, -1)
            y_factor = torch.sqrt(torch.sum(y ** 2)) + eps
            normalized_y = y / y_factor

        proj_loss = torch.sum(normalized_x * normalized_y) / n
        return proj_loss
    
class GaussianBlur:
    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class _MedianFilter:
    def __init__(self, size=3):
        self.size = size
    
    def __call__(self, img):
        blurred = img.filter(ImageFilter.MedianFilter(size=self.size))
        return blurred 
cnt=0
def save_all(original, advs, coarse, fine, gt, adv_pred, coarse_pred, fine_pred):
    global cnt
    coarse_lst = torch.unbind(coarse, dim=0)
    fine_lst = torch.unbind(fine, dim=0)
    original_lst = torch.unbind(original, dim=0)
    advs_lst = torch.unbind(advs, dim=0)
    coarse_pil = [tf.to_pil_image(coarse) for coarse in coarse_lst]
    fine_pil = [tf.to_pil_image(fine) for fine in fine_lst]
    original_pil = [tf.to_pil_image(o) for o in original_lst]
    advs_pil = [tf.to_pil_image(adv) for adv in advs_lst]

    for i, (o, a, c, f) in enumerate(zip(original_pil, advs_pil, coarse_pil, fine_pil)):
        o.save(os.path.join(result_dir, '{}_original_'.format(cnt)+gt[i]+'.png'))
        a.save(os.path.join(result_dir, '{}_adv_'.format(cnt)+adv_pred[i]+'.png'))
        c.save(os.path.join(result_dir, '{}_coarse_'.format(cnt)+coarse_pred[i]+'.png'))
        f.save(os.path.join(
            result_dir, '{}_fine_'.format(cnt)+fine_pred[i]+'.png'))
        cnt += 1
        