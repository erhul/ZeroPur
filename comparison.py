from robustbench.data import load_cifar10, load_cifar100
from robustbench.utils import load_model
# import foolbox as fb
import torch
import numpy as np
import random
import torchvision.transforms as transforms
from autoattack import AutoAttack
from torchvision import datasets
from tqdm import tqdm


def set_loader(data_folder='', batch_size=512, num_workers=8):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_dataset = datasets.CIFAR10(
        root=data_folder,
        transform=transform,
        download=True,
        train=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return val_loader

def eval_autoattack(model, x_test, y_test):
    adversary = AutoAttack(model, norm='Linf', eps=8/255.,
                           version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.n_restarts = 1
    x_adv = adversary.run_standard_evaluation(x_test, y_test)
    # x_test = torch.cat([x for (x, y) in data_loader], dim=0)
    # y_test = torch.cat([y for (x, y) in data_loader], dim=0)
    # x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=512)

def eval_standard(model, data_loader):
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    acc = None
    for x_test, y_test in tqdm(data_loader):
        x_test, y_test = x_test.cuda(), y_test.cuda()
        _, advs, success = fb.attacks.LinfPGD(steps=20, abs_stepsize=2/255.,)(fmodel, x_test, y_test, epsilons=[8/255],)
        # _, advs, success = fb.attacks.L2PGD(steps=20, abs_stepsize=15/255.,)(fmodel, x_test, y_test, epsilons=[0.5],)
        acc = success if acc is None else torch.cat([acc, success], dim=-1)
    print('Robust accuracy: {:.2%}'.format(1 - acc.float().mean()))

if __name__ == '__main__':
    # data_loader = set_loader()
    x_test, y_test = load_cifar100()
    print(x_test.shape, y_test.shape)
    # model = load_model(model_name='Wang2023Better_WRN-28-10', 
    #                    dataset='cifar10', threat_model='Linf').cuda()
    # model = load_model(model_name='Xu2023Exploring_WRN-28-10',
    #                      dataset='cifar10', threat_model='Linf').cuda()
    # model = load_model(model_name='Pang2022Robustness_WRN28_10',
    #                    dataset='cifar10', threat_model='Linf').cuda()
    # model = load_model(model_name='Rade2021Helper_R18_ddpm',
    #                    dataset='cifar100', threat_model='Linf').cuda()
    # model = load_model(model_name='Rebuffi2021Fixing_R18_ddpm',
    #                    dataset='cifar100', threat_model='Linf').cuda()
    # model = load_model(model_name='Cui2023Decoupled_WRN-34-10',
    #                    dataset='cifar100', threat_model='Linf').cuda()
    model = load_model(model_name='Chen2024Data_WRN_34_10',
                       dataset='cifar100', threat_model='Linf').cuda()
    # model = load_model(model_name='Gowal2021Improving_R18_ddpm_100m',
    #                dataset='cifar10', threat_model='Linf').cuda()
    # model = load_model(model_name='Sehwag2021Proxy_R18',
    #                dataset='cifar10', threat_model='Linf')
    # model = load_model(model_name='Rade2021Helper_R18_extra',
    #                dataset='cifar10', threat_model='Linf')
    # model = load_model(model_name='Addepalli2022Efficient_RN18',
    #                dataset='cifar10', threat_model='Linf')
    # model = load_model(model_name='Gowal2021Improving_28_10_ddpm_100m',
    #                dataset='cifar10', threat_model='Linf')
    # model = load_model(model_name='Pang2022Robustness_WRN28_10',
    #                dataset='cifar10', threat_model='Linf')
    # model = load_model(model_name='Gowal2020Uncovering_28_10_extra',
    #                dataset='cifar10', threat_model='Linf')
    # model = load_model(model_name='Rade2021Helper_extra',
    #                dataset='cifar10', threat_model='Linf')
    # model = load_model(model_name='Rade2021Helper_R18_ddpm',
    #                dataset='cifar10', threat_model='L2')
    # model = load_model(model_name='Rebuffi2021Fixing_R18_cutmix_ddpm',
    #                dataset='cifar10', threat_model='L2')
    # model = load_model(model_name='Sehwag2021Proxy_R18',
    #                dataset='cifar10', threat_model='L2')
    # model = load_model(model_name='Rice2020Overfitting',
    #                dataset='cifar10', threat_model='L2')
    # model = load_model(model_name='Augustin2020Adversarial_34_10',
    #                dataset='cifar10', threat_model='L2')
    # model = load_model(model_name='Sehwag2021Proxy',
    #                dataset='cifar10', threat_model='L2')
    # model = load_model(model_name='Augustin2020Adversarial_34_10_extra',
    #                dataset='cifar10', threat_model='L2')
    # model = load_model(model_name='Wu2020Adversarial',
    #                dataset='cifar10', threat_model='L2')
    # model = load_model(model_name='Wang2023Better_WRN-28-10',
    #                dataset='cifar10', threat_model='L2')
    # model = load_model(model_name='Rade2021Helper_R18_ddpm',
    #                dataset='cifar100', threat_model='Linf')
    # model = load_model(model_name='Addepalli2021Towards_PARN18',
    #                dataset='cifar100', threat_model='Linf')
    # model = load_model(model_name='Addepalli2022Efficient_RN18',
    #                dataset='cifar100', threat_model='Linf')
    # model = load_model(model_name='Rebuffi2021Fixing_28_10_cutmix_ddpm',
    #                dataset='cifar100', threat_model='Linf')
    # model = load_model(model_name='Pang2022Robustness_WRN28_10',
    #                dataset='cifar100', threat_model='Linf')
    # model = load_model(model_name='Addepalli2022Efficient_WRN_34_10',
    #                dataset='cifar100', threat_model='Linf')
    # model = load_model(model_name='Cui2020Learnable_34_10_LBGAT6',
    #                dataset='cifar100', threat_model='Linf')
    # model = load_model(model_name='Jia2022LAS-AT_34_10',
    #                dataset='cifar100', threat_model='Linf')
    # eval_standard(model, data_loader)
    # eval_autoattack(model, data_loader)
    eval_autoattack(model, x_test, y_test)