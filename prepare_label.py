import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def get_classes_map():
    result_map = {}
    with open('./map_clsloc.txt', encoding='utf-8') as f:
        for line in f:
            contents = line.strip().split(' ')
            cls_idx, cls_name = contents[0], contents[-1]
            result_map[cls_idx] = cls_name
    return result_map

