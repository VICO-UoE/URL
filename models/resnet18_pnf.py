import torch.nn as nn
import torch
from functools import partial

from models.model_utils import CosineClassifier


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CatFilm(nn.Module):
    """Film layer that performs per-channel affine transformation."""
    def __init__(self, planes):
        super(CatFilm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, planes))
        self.beta = nn.Parameter(torch.zeros(1, planes))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        return gamma * x + beta


class BasicBlockFilm(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockFilm, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.film1 = CatFilm(planes)
        self.film2 = CatFilm(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.film1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.film2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, classifier=None, num_classes=None,
                 dropout=0.0, global_pool=True):
        super(ResNet, self).__init__()
        self.initial_pool = False
        self.film_normalize = CatFilm(3)
        inplanes = self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.outplanes = 512

        # handle classifier creation
        if num_classes is not None:
            if classifier == 'linear':
                self.cls_fn = nn.Linear(self.outplanes, num_classes)
            elif classifier == 'cosine':
                self.cls_fn = CosineClassifier(self.outplanes, num_classes)

        # initialize everything
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.embed(x)
        x = self.cls_fn(x)
        return x

    def embed(self, x, squeeze=True, param_dict=None):
        """Computing the features"""
        x = self.film_normalize(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x.squeeze()

    def get_state_dict(self):
        """Outputs the state elements that are domain-specific"""
        return {k: v for k, v in self.state_dict().items()
                if 'film' in k or 'cls' in k or 'running' in k}

    def get_parameters(self):
        """Outputs only the parameters that are domain-specific"""
        return [v for k, v in self.named_parameters()
                if 'film' in k or 'cls' in k]


def resnet18(pretrained=False, pretrained_model_path=None, **kwargs):
    """
        Constructs a FiLM adapted ResNet-18 model.
    """
    model = ResNet(BasicBlockFilm, [2, 2, 2, 2], **kwargs)

    # loading shared convolutional weights
    if pretrained_model_path is not None:
        device = model.get_parameters()[0].device
        ckpt_dict = torch.load(pretrained_model_path, map_location=device)['state_dict']
        shared_state = {k: v for k, v in ckpt_dict.items() if 'cls' not in k}
        model.load_state_dict(shared_state, strict=False)
        print('Loaded shared weights from {}'.format(pretrained_model_path))
    return model
