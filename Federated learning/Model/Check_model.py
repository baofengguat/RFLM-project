from .BotNet.BotNetBottlent import *
from config import load_config
from torchvision.models import resnet34,resnet18,vgg16,densenet121,mobilenet_v2
from .DBB_ResNet.convnet_utils import *

args = load_config()

def Check_model(modelSelect):
    if modelSelect == 'Botnet18':
        model = Botnet18(pretrained=True, NumClass=2)
        print('Model use:->> Botnet18')

    elif modelSelect == 'Botnet34':
        model = Botnet34(pretrained=True, NumClass=2)
        print('Model use:->> Botnet34')

    elif modelSelect == 'ResNet18':
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
        print('Model use:->> ResNet18')

    elif modelSelect == 'ResNet34':
        model = resnet34(pretrained=True)
        model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
        print('Model use:->> ResNet34')

    elif modelSelect == 'VGG16':
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)
        print('Model use:->> VGG16')

    elif modelSelect == 'densenet121':
        model = densenet121(pretrained=True)
        model.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)
        print('Model use:->> densenet121')

    elif modelSelect == 'DBB_ResNet18':
        switch_deploy_flag(False)
        switch_conv_bn_impl('DBB')
        model = build_model('ResNet-18')
        print('Model use:->> DBB_ResNet18')

    elif modelSelect == 'DBB_ResNet50':
        switch_deploy_flag(False)
        switch_conv_bn_impl('DBB')
        model = build_model('ResNet-50')
        print('Model use:->> DBB_ResNet50')

    elif modelSelect == 'ResNet18':
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
        print('Model use:->> ResNet18')

    else:
        model = mobilenet_v2(pretrained=True)
        model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
        print('error:-->>Model selection failed\nModel use: :->> mobilenet')

    return model

