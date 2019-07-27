# python3.7
"""Contains a collection of supported models."""

import torchvision.models as models

__all__ = ['get_model']

MODELS = {
    ### CLASSIFICATION ###
    # AlexNet
    'alexnet': models.alexnet,
    # VGG
    'vgg11': models.vgg11,
    'vgg11_bn': models.vgg11_bn,
    'vgg13': models.vgg13,
    'vgg13_bn': models.vgg13_bn,
    'vgg16': models.vgg16,
    'vgg16_bn': models.vgg16_bn,
    'vgg19': models.vgg19,
    'vgg19_bn': models.vgg19_bn,
    # Inception
    'googlenet': models.googlenet,
    'incepetion_v3': models.inception_v3,
    # ResNet
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    # ResNext
    'resnext50': models.resnext50_32x4d,
    'resnext101': models.resnext101_32x8d,
    # DenseNet
    'densenet121': models.densenet121,
    'densenet169': models.densenet169,
    'densenet161': models.densenet161,
    'densenet201': models.densenet201,
    # SqueezeNet
    'squeezenet': models.squeezenet1_0,
    'squeezenet_fast': models.squeezenet1_1,
    # MobileNet
    'mobilenet': models.mobilenet_v2,
    # ShuffleNet
    'shufflenet_0_5': models.shufflenet_v2_x0_5,
    'shufflenet_1_0': models.shufflenet_v2_x1_0,
    'shufflenet_1_5': models.shufflenet_v2_x1_5,
    'shufflenet_2_0': models.shufflenet_v2_x2_0,

    ### SEGMENTATION ###
    # FCN
    'fcn_resnet50': models.segmentation.fcn_resnet50,
    'fcn_resnet101': models.segmentation.fcn_resnet101,
    # DeepLab
    'deeplab_resnet50': models.segmentation.deeplabv3_resnet50,
    'deeplab_resnet101': models.segmentation.deeplabv3_resnet101,

    ### DETECTION ###
    # Faster R-CNN
    'fasterrcnn_resnet50': models.detection.fasterrcnn_resnet50_fpn,
    # Mask R-CNN
    'maskrcnn_resnet50': models.detection.maskrcnn_resnet50_fpn,
    # Keypoint R-CNN
    'keypoint_resnet50': models.detection.keypointrcnn_resnet50_fpn,
}


def get_model(model_name, use_pretrain=False, **kwargs):
  """Gets model by name and loads pre-trained model if needed."""
  model_name = model_name.lower()
  try:
    model = MODELS[model_name](pretrained=use_pretrain, **kwargs)
  except KeyError:
    raise ValueError(f'Model `{model_name}` is not supported!\n'
                     f'Please choose from {list(MODELS)}.')
  return model
