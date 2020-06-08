dependencies = ['torch','torchvision']

import network
import torch

from network import deeplabv3_resnet50 as _deeplabv3_resnet50
from network import deeplabv3plus_resnet50 as _deeplabv3plus_resnet50
from network import deeplabv3_resnet101 as _deeplabv3_resnet101
from network import deeplabv3plus_resnet101 as _deeplabv3plus_resnet101
from network import deeplabv3_mobilenet as _deeplabv3_mobilenet
from network import deeplabv3plus_mobilenet as _deeplabv3plus_mobilenet

def deeplabv3plus_resnet50(pretrained=False,**kwargs):
  model=_deeplabv3plus_resnet50(**kwargs)
  if pretrained:
    checkpoint = 'https://github.com/joemarshall/DeepLabV3Plus-Pytorch/releases/download/pretrained_1.0/best_deeplabv3plus_resnet50_voc_os16.pth'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False)["model_state"])
  return model
  
def deeplabv3_resnet50(pretrained=False,**kwargs):
  model=_deeplabv3_resnet50(**kwargs)
  if pretrained:
    checkpoint = 'https://github.com/joemarshall/DeepLabV3Plus-Pytorch/releases/download/pretrained_1.0/best_deeplabv3_resnet50_voc_os16.pth'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False)["model_state"])
  return model

def deeplabv3_resnet101(pretrained=False,**kwargs):
  model=_deeplabv3_resnet101(**kwargs)
  if pretrained:
    checkpoint = 'https://github.com/joemarshall/DeepLabV3Plus-Pytorch/releases/download/pretrained_1.0/best_deeplabv3_resnet101_voc_os16.pth'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False)["model_state"])
  return model

def deeplabv3plus_resnet101(pretrained=False,**kwargs):
  model=_deeplabv3plus_resnet101(**kwargs)
  if pretrained:
    checkpoint = 'https://github.com/joemarshall/DeepLabV3Plus-Pytorch/releases/download/pretrained_1.0/best_deeplabv3plus_resnet101_voc_os16.pth'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False)["model_state"])
  return model

def deeplabv3_mobilenet(pretrained=False,**kwargs):
  model=_deeplabv3_mobilenet(**kwargs)
  if pretrained:
    checkpoint = 'https://github.com/joemarshall/DeepLabV3Plus-Pytorch/releases/download/pretrained_1.0/best_deeplabv3_mobilenet_voc_os16.pth'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False)["model_state"])
  return model

def deeplabv3plus_mobilenet(pretrained=False,**kwargs):
  model=_deeplabv3plus_mobilenet(**kwargs)
  if pretrained:
    checkpoint = 'https://github.com/joemarshall/DeepLabV3Plus-Pytorch/releases/download/pretrained_1.0/best_deeplabv3plus_mobilenet_voc_os16.pth'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False)["model_state"])
  return model
