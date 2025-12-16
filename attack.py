import torch
import torch.nn as nn
import timm
import os
import time

import utils
import torchvision.transforms.functional as F

from autoattack import AutoAttack
from stadv_eot.attacks import StAdvAttack
from PIL import Image
# from torchattacks import *

def run():
    fmodel = timm.create_model('resnet50', pretrained=True)
    model = nn.Sequential(
        utils.Normalize(),
        fmodel,
    ).cuda().eval()
    
    val_data = utils.load_data(domain='imagenet', batch_size=128)
    x_val, y_val = next(iter(val_data))
    x_val, y_val = x_val.cuda(), y_val.cuda()

    x_adv_linf = run_autoattack(model, x_val, y_val, batch_size=128, norm='Linf')
    x_adv_l2 = run_autoattack(model, x_val, y_val, batch_size=128, norm='L2')
    x_adv_stadv = run_stadv(model, x_val, y_val, batch_size=128)

    linf_acc, linf_wrong = utils.get_accuracy(model, x_adv_linf, y_val, bs=128)
    l2_acc, l2_wrong = utils.get_accuracy(model, x_adv_l2, y_val, bs=128)
    stadv_acc, stadv_wrong = utils.get_accuracy(model, x_adv_stadv, y_val, bs=128)

    linf_adv = torch.unbind(x_adv_linf[linf_wrong], dim=0)
    linf_nat = torch.unbind(x_val[linf_wrong], dim=0)
    linf_adv_y = utils.get_pred(model, x_adv_linf[linf_wrong])
    linf_nat_y = utils.get_pred(model, x_val[linf_wrong])

    l2_adv = torch.unbind(x_adv_l2[l2_wrong], dim=0)
    l2_nat = torch.unbind(x_val[l2_wrong], dim=0)
    l2_adv_y = utils.get_pred(model, x_adv_l2[l2_wrong],)
    l2_nat_y = utils.get_pred(model, x_val[l2_wrong])

    stadv_adv = torch.unbind(x_adv_stadv[stadv_wrong], dim=0)
    stadv_nat = torch.unbind(x_val[stadv_wrong], dim=0)
    stadv_adv_y = utils.get_pred(model, x_adv_stadv[stadv_wrong])
    stadv_nat_y = utils.get_pred(model, x_val[stadv_wrong])
    
    for i, (adv, adv_y, nat, nat_y) in enumerate(zip(linf_adv, linf_adv_y, linf_nat, linf_nat_y)):
        os.makedirs('./results/{}/Linf'.format(i))
        adv_img, nat_img = F.to_pil_image(adv), F.to_pil_image(nat)
        adv_img.save(os.path.join('./results/{}/Linf'.format(i), 'adv_{}.png'.format(adv_y.item())))
        nat_img.save(os.path.join('./results/{}/Linf'.format(i), 'nat_{}.png'.format(nat_y.item())))
    print('Linf robust acc: {:.2%}, saved {} images'.format(linf_acc, len(linf_adv)))

    for i, (adv, adv_y, nat, nat_y) in enumerate(zip(stadv_adv, stadv_adv_y, stadv_nat, stadv_nat_y)):
        os.makedirs('./results/{}/L2'.format(i))
        adv_img, nat_img = F.to_pil_image(adv), F.to_pil_image(nat)
        adv_img.save(os.path.join('./results/{}/L2'.format(i), 'adv_{}.png'.format(adv_y.item())))
        nat_img.save(os.path.join('./results/{}/L2'.format(i), 'nat_{}.png'.format(nat_y.item())))
    print('L2 robust acc: {:.2%}, saved {} images'.format(l2_acc, len(l2_adv)))

    for i, (adv, adv_y, nat, nat_y) in enumerate(zip(l2_adv, l2_adv_y, l2_nat, l2_nat_y)):
        os.makedirs('./results/{}/stadv'.format(i))
        adv_img, nat_img = F.to_pil_image(adv), F.to_pil_image(nat)
        adv_img.save(os.path.join('./results/{}/stadv'.format(i), 'adv_{}.png'.format(adv_y.item())))
        nat_img.save(os.path.join('./results/{}/stadv'.format(i), 'nat_{}.png'.format(nat_y.item())))
    print('StAdv robust acc: {:.2%}, saved {} images'.format(stadv_acc, len(stadv_adv)))


def run_autoattack(model, x_val, y_val, batch_size, norm='Linf', return_labels=False):
    if norm == 'Linf':
        adversary = AutoAttack(model, norm='Linf', eps=8/255., 
                            version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    elif norm == 'L2':
        adversary = AutoAttack(model, norm='L2', eps=0.5, 
                            version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    else:
        raise NotImplementedError
    adversary.apgd.n_restarts = 1
    adversary.seed = 42

    x_adv = adversary.run_standard_evaluation(x_val, y_val, bs=batch_size, return_labels=return_labels)

    return x_adv
    
def run_stadv(model, x_val, y_val, batch_size):
    adversary = StAdvAttack(model, bound=1, num_iterations=100) # 0.07


    start_time = time.time()
    init_acc, _ = utils.get_accuracy(model, x_val, y_val, bs=batch_size)
    print('initial accuracy: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, time.time() - start_time))

    start_time = time.time()
    x_adv = adversary(x_val, y_val)
    robust_acc, _= utils.get_accuracy(model, x_adv, y_val, batch_size)
    print('robust accuracy: {:.2%}, time elapsed: {:.2f}s'.format(robust_acc, time.time() - start_time))

    return x_adv

if __name__ == '__main__':

    run()

    