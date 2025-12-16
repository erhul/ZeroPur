import torch
from torchvision.models import alexnet
from .models import AlexNetFeatureModel, CifarAlexNet, CifarPreActResNetFeatureModel, CifarWideResNetFeatureModel, CifarResNetFeatureModel
from torch.hub import load_state_dict_from_url


def get_self_lpips_model(model, arch='r18'):
    if arch == 'r18':
        return CifarPreActResNetFeatureModel(model)
    elif arch == 'wrn':
        return CifarWideResNetFeatureModel(model)
    elif arch == 'r50':
        return CifarResNetFeatureModel(model)
    elif arch =='vgg':
        return torch.nn.Identity()
    else:
        raise NotImplementedError


def get_lpips_model(model='alexnet_cifar', 
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    if model == 'alexnet':
        alexnet_model = alexnet(pretrained=True)
        lpips_model = AlexNetFeatureModel(alexnet_model).to(device)
    
    elif model == 'alexnet_cifar':
        alexnet_model = CifarAlexNet()
        lpips_model = AlexNetFeatureModel(alexnet_model).to(device)
        try:
            state = torch.load('data/checkpoints/alexnet_cifar.pt')
        except FileNotFoundError:
            state = load_state_dict_from_url(
                'https://perceptual-advex.s3.us-east-2.amazonaws.com/'
                'alexnet_cifar.pt',
                progress=True,
            )
        lpips_model.load_state_dict(state['model'])
        
    else:
        raise NotImplementedError
    
    lpips_model.eval()

    return lpips_model

