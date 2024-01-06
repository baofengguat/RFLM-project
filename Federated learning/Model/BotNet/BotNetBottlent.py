from torch import nn
from torchvision.models import resnet50,resnet34,resnet18
from .Botlayer import BottleStack
import sys
sys.path.append(r'.\DL_Platform\Model')

layer = BottleStack(
    dim = 64,
    fmap_size = 56,        # set specifically for imagenet's 224 x 224
    dim_out = 2048,
    proj_factor = 4,
    downsample = True,
    heads = 4,
    dim_head = 128,
    rel_pos_emb = True,
    activation = nn.ReLU()
)

def Botnet18(pretrained,NumClass):
    resnet = resnet18(pretrained=pretrained)
    backbone = list(resnet.children())
    model = nn.Sequential(
        *backbone[:4],
        layer,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1),
        nn.Linear(2048, NumClass)
    )
    return model

def Botnet34(pretrained,NumClass):
    resnet = resnet34(pretrained=pretrained)
    backbone = list(resnet.children())
    model = nn.Sequential(
        *backbone[:5],
        layer,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1),
        nn.Linear(2048, NumClass)
    )
    return model





