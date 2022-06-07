import torch.nn as nn
import torch
from models.model_utils import CosineClassifier
from config import args

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, classifier=None, num_classes=64,
                 dropout=0.0, global_pool=True):
        super(ResNet, self).__init__()
        self.initial_pool = False
        inplanes = self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2,
                               padding=2, bias=False)
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
        self.num_classes = num_classes
        # handle classifiers creation
        if num_classes is not None:
            cls_fn = []
            if classifier == 'linear':
                for num_class in num_classes:
                    cls_fn.append(nn.Linear(self.outplanes, num_class))
                self.cls_fn = nn.ModuleList(cls_fn)
            elif classifier == 'cosine':
                for num_class in num_classes:
                    cls_fn.append(CosineClassifier(self.outplanes, num_class))
                self.cls_fn = nn.ModuleList(cls_fn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, num_samples=None, kd=False):
        if kd:
            if num_samples is not None:
                embed_ = self.embed(x, num_samples)
                embed = list(torch.split(embed_, num_samples))
            else:
                embed = list(torch.split(self.embed(x, num_samples), int(x.size(0)/len(self.num_classes))))
            x = self.dropout(torch.cat(embed, dim=0))
        else:
            x = self.embed(x, num_samples)
            x = self.dropout(x)
        if num_samples is None:
            x = list(torch.split(x, int(x.size(0)/len(self.num_classes))))
            out = []
            for t in range(len(self.num_classes)):
                out.append(self.cls_fn[t](x[t]))
        else:
            x = list(torch.split(x, num_samples))
            out = []
            for t in range(len(self.num_classes)):
                out.append(self.cls_fn[t](x[t]))
        if kd:
            return out, embed
        else:
            return out

    def embed(self, x, num_samples=None, param_dict=None):
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
        # return x.squeeze()
        return x.flatten(1)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.named_parameters()]


def resnet18(pretrained=False, pretrained_model_path=None, **kwargs):
    """
        Constructs a ResNet-18 multi-domain learning model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained_model_path is not None:
        device = model.get_parameters()[0].device
        ckpt_dict = torch.load(pretrained_model_path, map_location=device)['state_dict']
        shared_state = {k: v for k, v in ckpt_dict.items() if 'cls' not in k}
        model.load_state_dict(shared_state, strict=False)
        print('Loaded shared weights from {}'.format(pretrained_model_path))
    return model
