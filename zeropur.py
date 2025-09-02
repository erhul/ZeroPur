# -*- coding: utf-8 -*-
import sys


sys.path.append('./..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import ILAProjLoss
from peceptual.misc import get_self_lpips_model
from peceptual.distances import LPIPSDistance


class ZeroPur(nn.Module):
    def __init__(self, model, transform, normalizer=nn.Identity(), arch='r18', norm='Linf'):
        super().__init__()
        self.model = model
        self.transform = transform
        self.normalizer = normalizer
        self.norm = norm
        self.lpips_dist = LPIPSDistance(get_self_lpips_model(model, arch=arch))
        self.arch = arch

    def forward_batch(self, dataloader, two_phase=True, lam1=1e-4, lam2=1, device=torch.device('cuda'), **kwargs):
        fine_res, coarse_res, y_res = [], [], []
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x_fine, x_coarse = self.forward(x, two_phase=two_phase, arch=self.arch, lam1=lam1, lam2=lam2, *kwargs)
            fine_res.append(x_fine)
            y_res.append(y)
            if two_phase:
                coarse_res.append(x_coarse)
        fine_res = torch.cat(fine_res, dim=0)
        y_res = torch.cat(y_res, dim=0)
        if two_phase:
            coarse_res = torch.cat(coarse_res, dim=0)
            return fine_res, coarse_res, y_res
        else:
            return fine_res, y_res


    def forward(self, inputs, two_phase=True, lam1=1e-4, lam2=1, **kwargs, ):
        coarse = self.coarse_shifting(inputs, **kwargs)
        if two_phase:
            fine = self.fine_alignment(original=inputs, coarse=coarse, lam1=lam1, lam2=lam2, **kwargs)
            return fine, coarse
        else:
            return None, coarse
    
    def coarse_shifting(self, inputs, rand=False, eps=10 / 255., num_steps=10, step_size=1 / 255., **kwargs):
        x = inputs.clone()
        if rand:
            x = x + torch.zeros_like(x).uniform_(-eps, eps)

        if self.arch  == 'r18':
            insider = nn.Sequential(
                self.normalizer, *list(self.model.children())[:5], nn.AvgPool2d(4), nn.Flatten(1),
                ).to(x.device).eval()
        elif self.arch  == 'r50':
            insider = nn.Sequential(
                self.normalizer, *list(self.model.children())[:-2], self.model.global_pool, nn.Flatten(1),
                ).to(x.device).eval()
        elif self.arch  == 'wrn':
            insider = nn.Sequential(
                self.normalizer, *list(self.model.children())[:-1], nn.Flatten(1),
            ).to(x.device).eval()
        elif self.arch =='vgg':
            insider = nn.Sequential(
                self.normalizer, self.model.features, self.model.pre_logits
            ).to(x.device).eval()
        else:
            raise NotImplementedError

        # TODO:
        step_size = eps / num_steps

        momentum = torch.zeros_like(inputs, device=inputs.device)
        # x_old = x.clone()
        for i in range(num_steps):
            x.requires_grad_(True)
            aug_x = self.transform(x.clone().detach())
            with torch.enable_grad():
                features = insider(x)
                aug_features = insider(aug_x)

                loss = -1 * F.cosine_similarity(features, aug_features).mean()
                
            grad = torch.autograd.grad(loss, [x])[0].detach()

            grad_norm = torch.norm(grad, p=1, dim=(1, 2, 3)).view(-1, 1, 1, 1) # grad_norm = torch.norm(grad, p=1)
            grad /= grad_norm
            grad += momentum * 1.0
            momentum = grad

            with torch.no_grad():
                x = x.detach()
                # grad2 = x - x_old
                # x_old = x.clone()
                # a = 0.5 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x1 = x - step_size * torch.sign(grad)
                    x1 = torch.clamp(x1, min=inputs - eps, max=inputs + eps)
                    x1 = torch.clamp(x1, 0.0, 1.0)

                    # x1 = x + (x1 - x) * a + grad2 * (1 - a)
                    # x1 = torch.clamp(x1, min=inputs - eps, max=inputs + eps)
                    # x1 = torch.clamp(x1, 0.0, 1.0)

                elif self.norm == 'L2':
                    grad_norms = torch.norm(grad.view(x.size(0), -1), p=2, dim=1) + 1e-12
                    grad /= grad_norms.view(x.size(0), 1, 1, 1)
                    x1 = x - step_size * grad

                    delta = x1 - inputs
                    delta_norms = torch.norm(delta.view(x.size(0), -1), p=2, dim=1)
                    factor = eps / delta_norms
                    factor = torch.min(factor, torch.ones_like(delta_norms))
                    delta = delta * factor.view(-1, 1, 1, 1)
                    x1 = torch.clamp(inputs + delta, min=0.0, max=1.0)

                    # x1 = x + (x1 - x) * a + grad2 * (1 - a)
                    # delta = x1 - inputs
                    # delta_norms = torch.norm(delta.view(x.size(0), -1), p=2, dim=1)
                    # factor = eps / delta_norms
                    # factor = torch.min(factor, torch.ones_like(delta_norms))
                    # delta = delta * factor.view(-1, 1, 1, 1)
                    # x1 = torch.clamp(inputs + delta, min=0.0, max=1.0)
                x = x1 + 0.

        return x
    
    def fine_alignment(self, original, coarse, gamma=0.5, eps=10 / 255., ila_step=50, lam1=1e-4, lam2=1, **kwargs):
        fine = original.clone()
        step_size = eps / ila_step
       
        for _ in range(ila_step):
            loss = []
            fine.requires_grad_(True)
            with torch.enable_grad():
                for stage in range(3, 5): #torch.Size([128, 256, 8, 8]),torch.Size([128, 512, 4, 4]),
                    if self.arch == 'r18':
                        insider = nn.Sequential(self.normalizer, *list(self.model.children())[:stage+1]).to(fine.device)
                    elif self.arch == 'r50':
                        insider = nn.Sequential(self.normalizer, *list(self.model.children())[:stage-6]).to(fine.device)
                    elif self.arch == 'wrn':
                        insider = nn.Sequential(self.normalizer, *list(self.model.children())[:stage]).to(fine.device)
                    else:
                        raise NotImplementedError

                    with torch.no_grad():
                        latent_coarse = insider(coarse)
                        latent_original = insider(original)
                    latent_current = insider(fine)

                    loss.append(-ILAProjLoss()(latent_coarse, latent_current, latent_original, 0.0))

                ila_loss = sum(loss)
                lpips_distance = self.lpips_dist(fine, original).mean()
                loss_iter = lam1 * ila_loss + lam2 * lpips_distance

            input_grad = torch.autograd.grad(loss_iter, fine, create_graph=False)[0]

            with torch.no_grad():
                fine = fine.detach()
                if self.norm == 'Linf':
                    fine = fine - step_size * torch.sign(input_grad)
                    fine = torch.clamp(fine, min=original - eps, max=original + eps)
                    fine = torch.clamp(fine, min=0.0, max=1.0)

                elif self.norm == 'L2':
                    input_grad_norms = torch.norm(
                                input_grad.view(fine.size(0), -1), p=2, dim=1) + 1e-10
                    input_grad = input_grad / input_grad_norms.view(fine.size(0), 1, 1, 1)
                    fine = fine - step_size * input_grad

                    delta = fine - original
                    delta_norms = torch.norm(
                        delta.view(fine.size(0), -1), p=2, dim=1)
                    factor = eps / delta_norms
                    factor = torch.min(factor, torch.ones_like(delta_norms))
                    delta = delta * factor.view(-1, 1, 1, 1)
                    fine = torch.clamp(original + delta, min=0, max=1)

        return fine