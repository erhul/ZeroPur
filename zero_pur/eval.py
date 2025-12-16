import sys


sys.path.append('./..')
from prepare_label import get_classes_map
import time
import timm
import torch
import numpy as np
import torch.nn as nn
from attack import run_autoattack, run_difgsm
from timm.utils import AverageMeter, accuracy
import torch.nn.functional as F
import helper
from zeropur import ZeroPur

from models.preact_resnet import PreActResNet18
from models import wideresnet
from bpda.bpda import BPDAPurifierWrapper, apgd_train, bpda, pgd_linf
from bpda.auxiliary_aware import corase_aware, fine_aware
from torchattacks import DIFGSM



def eval_difgsm(type='baseline', bs=512, arch='r18'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if arch == 'r18':
        classifier = PreActResNet18()
        ckpt = torch.load('checkpoint/resnet18_cifar10_{}/ckpt.pth'.format(type))
    elif arch == 'wrn':
        classifier = wideresnet()
        ckpt = torch.load('checkpoint/wrn2810_cifar10_{}/ckpt.pth'.format(type))
    classifier.load_state_dict(ckpt['net'])
    fmodel = nn.Sequential(
        helper.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        classifier,
    ).to(device)

    classifier.eval()
    fmodel.eval()

    val_data = helper.load_data(domain='cifar10')
    x_val, y_val = next(iter(val_data))

    adversary = DIFGSM(fmodel)
    advs = helper.run_difgsm(adversary, x_val, y_val, bs=bs)

    purifier = ZeroPur(classifier, transform=helper.WeakTransform('median' if type == 'none' else 'gaussian'), 
                       normalizer=helper.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                       arch=arch)

    start_time = time.time()
    fine, coarse = helper.run_purifier(purifier, advs, eps=10/255, arch=arch, bs=bs)
    acc_fine, _ = helper.get_accuracy(fmodel, fine, y_val, bs=bs)
    acc_coarse, _ = helper.get_accuracy(fmodel, coarse, y_val, bs=bs)
    print('coarse accuracy: {:.2%}, fine accuracy: {:.2%}, time elapsed: {:.2f}s'.format(
        acc_coarse, acc_fine, time.time() - start_time))

    start_time = time.time()
    nat_fine, nat_coarse = helper.run_purifier(purifier, x_val, eps=10/255, arch=arch, bs=bs)
    acc_nat_fine, _ = helper.get_accuracy(fmodel, nat_fine, y_val, bs=bs)
    acc_nat_coarse, _ = helper.get_accuracy(fmodel, nat_coarse, y_val, bs=bs)
    print('natural coarse accuracy: {:.2%}, natural fine accuracy: {:.2%}, time elapsed: {:.2f}s'.format(
        acc_nat_coarse, acc_nat_fine, time.time() - start_time))

def eval_cifar(type='baseline', bs=512, arch='r18'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if arch == 'r18':
        classifier = PreActResNet18()
        ckpt = torch.load('checkpoint/resnet18_cifar10_{}/ckpt.pth'.format(type))
    elif arch == 'wrn':
        classifier = wideresnet()
        ckpt = torch.load('checkpoint/wrn2810_cifar10_{}/ckpt.pth'.format(type))
    classifier.load_state_dict(ckpt['net'])
    fmodel = nn.Sequential(
        helper.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        classifier,
    ).to(device)

    classifier.eval()
    fmodel.eval()

    val_data = helper.load_data(domain='cifar10')
    x_val, y_val = next(iter(val_data))
    # x_val, y_val = x_val.to(device), y_val.to(device)

    advs = run_autoattack(fmodel, x_val, y_val, batch_size=bs)

    purifier = ZeroPur(classifier, transform=helper.WeakTransform('median' if type == 'none' else 'gaussian'), 
                       normalizer=helper.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                       arch=arch)
    
    start_time = time.time()
    fine, coarse = helper.run_purifier(purifier, advs, eps=10/255, arch=arch, bs=bs)
    acc_fine, _ = helper.get_accuracy(fmodel, fine, y_val, bs=bs)
    acc_coarse, _ = helper.get_accuracy(fmodel, coarse, y_val, bs=bs)
    print('coarse accuracy: {:.2%}, fine accuracy: {:.2%}, time elapsed: {:.2f}s'.format(
        acc_coarse, acc_fine, time.time() - start_time))

    start_time = time.time()
    nat_fine, nat_coarse = helper.run_purifier(purifier, x_val, eps=10/255, arch=arch, bs=bs)
    acc_nat_fine, _ = helper.get_accuracy(fmodel, nat_fine, y_val, bs=bs)
    acc_nat_coarse, _ = helper.get_accuracy(fmodel, nat_coarse, y_val, bs=bs)
    print('natural coarse accuracy: {:.2%}, natural fine accuracy: {:.2%}, time elapsed: {:.2f}s'.format(
        acc_nat_coarse, acc_nat_fine, time.time() - start_time))
    
def eval_cifar100(type='baseline', bs=512, arch='r18'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if arch == 'r18':
        classifier = PreActResNet18()
        ckpt = torch.load('checkpoint/resnet18_cifar100_{}/ckpt.pth'.format(type))
    elif arch == 'wrn':
        classifier = wideresnet()
        ckpt = torch.load('checkpoint/wrn2810_cifar100_{}/ckpt.pth'.format(type))
    classifier.load_state_dict(ckpt['net'])
    fmodel = nn.Sequential(
        helper.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        classifier,
    ).to(device)

    classifier.eval()
    fmodel.eval()

    val_data = helper.load_data(domain='cifar100')
    x_val, y_val = next(iter(val_data))
    # x_val, y_val = x_val.to(device), y_val.to(device)

    advs = run_autoattack(fmodel, x_val, y_val, batch_size=bs)

    purifier = ZeroPur(classifier, transform=helper.WeakTransform('median' if type == 'none' else 'gaussian'), 
                       normalizer=helper.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)), arch=arch)
    
    start_time = time.time()
    fine, coarse = helper.run_purifier(purifier, advs, eps=10/255., arch=arch, bs=bs)
    acc_fine, _ = helper.get_accuracy(fmodel, fine, y_val, bs=bs)
    acc_coarse, _ = helper.get_accuracy(fmodel, coarse, y_val, bs=bs)
    print('coarse accuracy: {:.2%}, fine accuracy: {:.2%}, time elapsed: {:.2f}s'.format(
        acc_coarse, acc_fine, time.time() - start_time))

    start_time = time.time()
    nat_fine, nat_coarse = helper.run_purifier(purifier, x_val, eps=10/255., arch=arch, bs=bs)
    acc_nat_fine, _ = helper.get_accuracy(fmodel, nat_fine, y_val, bs=bs)
    acc_nat_coarse, _ = helper.get_accuracy(fmodel, nat_coarse, y_val, bs=bs)
    print('natural coarse accuracy: {:.2%}, natural fine accuracy: {:.2%}, time elapsed: {:.2f}s'.format(
        acc_nat_coarse, acc_nat_fine, time.time() - start_time))
    
    
def eval_batch_imagenet():

    classes_map = get_classes_map()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # classifier = timm.create_model('resnet50', pretrained=True)
    classifier = timm.create_model('vgg19_bn', pretrained=True)
    fmodel = nn.Sequential(helper.Normalize(), classifier,).to(device)
    classifier.eval()
    fmodel.eval()

    purifier = ZeroPur(classifier, transform=helper.WeakTransform(), normalizer=helper.Normalize(), arch='vgg')
    tf = helper.WeakTransform()
    val_data, cls_idx = helper.load_data(domain='imagenet')
    avg_fine, avg_coarse = AverageMeter(), AverageMeter()
    avg_nat_fine, avg_nat_coarse = AverageMeter(), AverageMeter()
    avg_blurred, avg_natural = AverageMeter(), AverageMeter()

    for step, (images, labels) in enumerate(val_data):
        start_time = time.time()
        images, labels = images.to(device), labels.to(device)
        advs = run_autoattack(fmodel, images, labels, batch_size=images.shape[0])

        blurred = tf(advs)

        # fine, coarse = helper.run_purifier(purifier, advs, eps=6/255., arch='r50', bs=images.shape[0])
        coarse = helper.run_coarse_purifier(purifier, advs, eps=6/255., arch='vgg', bs=images.shape[0])
        # nat_fine, nat_coarse = helper.run_purifier(purifier, images, eps=6/255., arch='r50', bs=images.shape[0])

        acc_coarse = accuracy(fmodel(coarse), labels)
        acc_blurred = accuracy(fmodel(blurred), labels)
        acc_natural= accuracy(fmodel(images), labels)
        # acc_fine, acc_coarse = accuracy(fmodel(fine), labels), accuracy(fmodel(coarse), labels)
        # acc_nat_fine, acc_nat_coarse = accuracy(fmodel(nat_fine), labels), accuracy(fmodel(nat_coarse), labels)
        
        avg_natural.update(acc_natural[0].item(), images.shape[0])
        avg_blurred.update(acc_blurred[0].item(), images.shape[0])
        # avg_fine.update(acc_fine[0].item(), images.shape[0])
        avg_coarse.update(acc_coarse[0].item(), images.shape[0])
        # avg_nat_fine.update(acc_nat_fine[0].item(), images.shape[0])
        # avg_nat_coarse.update(acc_nat_coarse[0].item(), images.shape[0])

        # print('step {}: current coarse acc {:.2f}, fine acc {:.2f}, nat coarse acc {:.2f}, nat fine acc {:.2f},' \
        #       'time elapsed: {:.2f}s'.format(
        #     step+1, avg_coarse.avg, avg_fine.avg, avg_nat_coarse.avg, avg_nat_fine.avg, time.time() - start_time
        # ))
        print('step {}: current coarse acc {:.2f}, blurred acc {:.2f}, natural acc {:.2f},' \
              'time elapsed: {:.2f}s'.format(
            step+1, avg_coarse.avg, avg_blurred.avg, avg_natural.avg, time.time() - start_time
        ))

        # coarse_logits, fine_logits = fmodel(coarse), fmodel(fine)
        # adv_logits = fmodel(advs)
        # coarse_idx = torch.argmax(coarse_logits, dim=-1).tolist()
        # fine_idx = torch.argmax(fine_logits, dim=-1).tolist()
        # adv_idx = torch.argmax(adv_logits, dim=-1).tolist()

        # coarse_pred = [classes_map[cls_idx[idx]] for idx in coarse_idx]
        # fine_pred = [classes_map[cls_idx[idx]] for idx in fine_idx]
        # adv_pred = [classes_map[cls_idx[idx]] for idx in adv_idx]
        # gt = [classes_map[cls_idx[idx]] for idx in labels.tolist()]

        # helper.save_all(images, advs, coarse, fine, gt, adv_pred, coarse_pred, fine_pred)
    
    # print('coarse accuracy: {:.2f}, fine accuracy: {:.2f}.'.format(
    #     avg_coarse.avg, avg_fine.avg,))
    # print('natural coarse accuracy: {:.2f}, natural fine accuracy: {:.2f}.'.format(
    #     avg_nat_coarse.avg, avg_nat_fine.avg,))
    print('coarse accuracy: {:.2f}, blurred accuracy: {:.2f}, natural accuracy: {:.2f}.'.format(
        avg_coarse.avg, avg_blurred.avg, avg_natural.avg))

    

if __name__ == '__main__':

    eval_cifar('none')
    eval_cifar('baseline')
    eval_cifar('all')


    # eval_difgsm('all')
