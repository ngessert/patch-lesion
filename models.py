import torch
import numbers
import numpy as np
import functools
import h5py
import math
from torchvision import models
import pretrainedmodels
import torch.nn.functional as F
import types
import torch


def inceptionv3(num_classes=1000, pretrained='imagenet'):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    """
    model = models.inception_v3(pretrained=True,aux_logits=True)
    #if pretrained is not None:
    #    settings = pretrained_settings['inceptionv3'][pretrained]
    #    model = load_pretrained(model, num_classes, settings)

    # Modify attributs
    model.last_linear = model.fc
    del model.fc

    def features(self, input):
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(input) # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x) # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x) # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2) # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x) # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x) # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2) # 35 x 35 x 192
        x = self.Mixed_5b(x) # 35 x 35 x 256
        x = self.Mixed_5c(x) # 35 x 35 x 288
        x = self.Mixed_5d(x) # 35 x 35 x 288
        x = self.Mixed_6a(x) # 17 x 17 x 768
        x = self.Mixed_6b(x) # 17 x 17 x 768
        x = self.Mixed_6c(x) # 17 x 17 x 768
        x = self.Mixed_6d(x) # 17 x 17 x 768
        x = self.Mixed_6e(x) # 17 x 17 x 768
        #if self.training and self.aux_logits:
        #    self._out_aux = self.AuxLogits(x) # 17 x 17 x 768
        x = self.Mixed_7a(x) # 8 x 8 x 1280
        x = self.Mixed_7b(x) # 8 x 8 x 2048
        x = self.Mixed_7c(x) # 8 x 8 x 2048
        return x

    def logits(self, features):
        x = torch.mean(torch.mean(features,2), 2) # 1 x 1 x 2048
        #x = F.dropout(x, training=self.training) # 1 x 1 x 2048
        x = x.view(x.size(0), -1) # 2048
        x = self.last_linear(x) # 1000 (num_classes)
        #if self.training and self.aux_logits:
        #    aux = self._out_aux
        #    self._out_aux = None
        #    return x, aux
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


model_map = {'Dense121' : models.densenet121(pretrained=True),
             'Dense121Nopre' : models.densenet121(pretrained=False),
             'Dense169' : models.densenet169(pretrained=True),
             'Dense161' : models.densenet161(pretrained=True),
             'Dense201' : models.densenet201(pretrained=True),
             'Resnet50' : pretrainedmodels.__dict__['resnet50'](num_classes=1000, pretrained='imagenet'),
             'Resnet101' : models.resnet101(pretrained=True),   
             'InceptionV3': inceptionv3(pretrained=True),# models.inception_v3(pretrained=True),
             'se_resnext50': pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet'),
             'se_resnext101': pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet'),
             'se_resnet50': pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet'),
               }

def getModel(model_name):
  """Returns a function for a model
  Args:
    mdlParams: dictionary, contains configuration
    is_training: bool, indicates whether training is active
  Returns:
    model: A function that builds the desired model
  Raises:
    ValueError: If model name is not recognized.
  """
  if model_name not in model_map:
    raise ValueError('Name of model unknown %s' % model_name)
  func = model_map[model_name]
  @functools.wraps(func)
  def model():
      return func
  return model