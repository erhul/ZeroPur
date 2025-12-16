import sys
sys.path.append('./..')

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from helper import ILAProjLoss, GaussianBlur, WeakTransform, NormalizedILAProjLoss
from peceptual.misc import get_lpips_model, get_self_lpips_model
from peceptual.distances import LPIPSDistance, SSIM
from peceptual.models import CifarPreActResNetFeatureModel
from zero_pur.models import PreActResNet18, wideresnet


def get_distance(cur, target):
    pass


class ZeroPur(nn.Module):
    def __init__(self, model, transform, normalizer=nn.Identity(), arch='r18', norm='Linf'):
        super().__init__()
        self.model = model
        self.transform = transform
        self.normalizer = normalizer
        self.norm = norm
        # self.lpips_dist = LPIPSDistance(get_lpips_model(model=lpips_model))
        self.lpips_dist = LPIPSDistance(get_self_lpips_model(model, arch=arch))
        self.ssim_dist = SSIM()
        self.arch = arch


    def forward(self, inputs, two_phase=True, **kwargs, ):
        coarse = self.coarse_shifting(inputs, **kwargs)
        if two_phase:
            fine = self.fine_alignment(original=inputs, coarse=coarse, **kwargs)
            return fine, coarse
        else:
            return None, coarse
    
    def coarse_shifting(self, inputs, rand=False, eps=10 / 255., num_steps=10, step_size=1 / 255., **kwargs):
        arch = kwargs['arch']
        
        if 'nat_x' in kwargs:
            nat_x = kwargs['nat_x']
            a_c_lst = []
            b_c_lst = []

        x = inputs.clone()
        if rand:
            x = x + torch.zeros_like(x).uniform_(-eps, eps)
        momentum = torch.zeros_like(inputs, device=inputs.device)
        x.requires_grad_(True)

        if self.arch  == 'r18':
            insider = nn.Sequential(
                self.normalizer, *list(self.model.children())[:5], nn.AvgPool2d(4), nn.Flatten(1),
                ).to(x.device)
        elif self.arch  == 'r50':
            insider = nn.Sequential(
                self.normalizer, *list(self.model.children())[:-2], self.model.global_pool, nn.Flatten(1),
                ).to(x.device)
        elif self.arch  == 'wrn':
            insider = nn.Sequential(
                self.normalizer, *list(self.model.children())[:-1], nn.Flatten(1),
            ).to(x.device)
        elif self.arch =='vgg':
            insider = nn.Sequential(
                self.normalizer, self.model.features, self.model.pre_logits
            ).to(x.device)
        else:
            raise NotImplementedError
        
        if 'nat_x' in kwargs:
            with torch.no_grad():
                    nat_features = insider(nat_x)

        for _ in range(num_steps):
            aug_x = self.transform(x.clone().detach())
            loss = 0.
            with torch.enable_grad():
                features = insider(x)
                aug_features = insider(aug_x)

                loss = -1 * F.cosine_similarity(features, aug_features).mean()
                
            grad = torch.autograd.grad(loss, [x], create_graph=False)[0]

            grad_norm = torch.norm(grad, p=1)
            grad.data /= grad_norm.data
            grad.data += momentum * 1.0
            momentum = grad.data

            if self.norm == 'Linf':
                x.data = x.data - step_size * torch.sign(grad.detach())
                x.data = torch.min(torch.max(x, inputs - eps), inputs + eps)
                x.data = torch.clamp(x, 0, 1)

            elif self.norm == 'L2':
                grad_norms = torch.norm(
                    grad.view(x.size(0), -1), p=2, dim=1) + 1e-10
                grad = grad / grad_norms.view(x.size(0), 1, 1, 1)
                x.data = x.detach() - step_size * grad

                delta = x - inputs
                delta_norms = torch.norm(delta.view(x.size(0), -1), p=2, dim=1)
                factor = eps / delta_norms
                factor = torch.min(factor, torch.ones_like(delta_norms))
                delta = delta * factor.view(-1, 1, 1, 1)
                x.data = torch.clamp(inputs + delta, min=0, max=1)

            if 'nat_x' in kwargs:
                a_c = F.cosine_similarity(features, nat_features).mean().item()
                b_c = F.cosine_similarity(aug_features, nat_features).mean().item()
                a_c_lst.append(a_c)
                b_c_lst.append(b_c)

        if 'nat_x' in kwargs:
            print('one batch completed, the distance btw a & c: {}, the distance btw b & c: {}'.format(a_c_lst, b_c_lst))
        return x
    
    def fine_alignment(self, original, coarse, gamma=0.5, eps=10 / 255., ila_step=50, **kwargs):
        arch = kwargs['arch']
        step_size = eps / ila_step
        fine = original.clone()
        fine.requires_grad_(True)

        for _ in range(ila_step):
            loss = []
            # fine.requires_grad_(True)
            for stage in range(3, 5):
                if self.arch  == 'r18':
                    insider = nn.Sequential(self.normalizer, *list(self.model.children())[:stage+1]).to(fine.device)
                elif self.arch  == 'r50':
                    insider = nn.Sequential(self.normalizer, *list(self.model.children())[:stage+1]).to(fine.device)
                elif self.arch  == 'wrn':
                    insider = nn.Sequential(self.normalizer, *list(self.model.children())[:stage]).to(fine.device)
                else:
                    raise NotImplementedError

                # insider.zero_grad() 
                with torch.no_grad():
                    latent_coarse = insider(coarse)
                    latent_original = insider(original)
                latent_current = insider(fine)

                loss.append(-ILAProjLoss()(latent_coarse, latent_current, latent_original, 0.0))

            lpips_distance = self.lpips_dist(fine, original).mean()
            ila_loss = sum(loss)
            # print('loss_iter: {}, lpips: {}'.format(ila_loss, lpips_distance))
            loss_iter = 1e-4 * ila_loss + lpips_distance
            # loss_iter = ila_loss

            input_grad = torch.autograd.grad(loss_iter, fine, create_graph=False)[0]

            if self.norm == 'Linf':
                fine.data = fine.data - step_size * torch.sign(input_grad)
                fine.data = torch.clamp(fine, min=original - eps, max=original + eps)
                fine.data = torch.clamp(fine, min=0, max=1)
                # del loss
                # The above code should do the same but allow for AutoAttack to work
                
                # loss_iter.backward()
                # input_grad = fine.grad.data
                # fine = fine.data - step_size * torch.sign(input_grad)
                #  # as fine <- fine.data whose requires_grad = False, now the fine.requires_grad = False
                # fine = torch.min(torch.max(fine, original - eps), original + eps)
                # fine = torch.clamp(fine, 0, 1)
            elif self.norm == 'L2':
                input_grad_norms = torch.norm(
                            input_grad.view(fine.size(0), -1), p=2, dim=1) + 1e-10
                input_grad = input_grad / input_grad_norms.view(fine.size(0), 1, 1, 1)
                fine.data = fine.data - step_size * input_grad

                delta = fine - original
                delta_norms = torch.norm(
                    delta.view(fine.size(0), -1), p=2, dim=1)
                factor = eps / delta_norms
                factor = torch.min(factor, torch.ones_like(delta_norms))
                delta = delta * factor.view(-1, 1, 1, 1)
                fine.data = torch.clamp(original + delta, min=0, max=1)
            
        return fine
    

if __name__ == '__main__':
    model = PreActResNet18()
    # lpips_model = CifarPreActResNetFeatureModel(model)
    # lpips_dist = LPIPSDistance(lpips_model)
    
    # x = torch.randn(10, 3, 32, 32)
    # y = torch.randn(10, 3, 32, 32)

    # d = lpips_dist(x, y)

    # model = wideresnet()
    print(list(model.children()))