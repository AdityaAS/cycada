import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .models import register_model

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.relu(out)

        return out

class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        out += shortcut
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, downblock, upblock, num_layers, n_classes):
        super(ResNet, self).__init__()

        self.transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ])

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_downlayer(downblock, 64, num_layers[0])
        self.layer2 = self._make_downlayer(downblock, 128, num_layers[1], stride=2)
        self.layer3 = self._make_downlayer(downblock, 256, num_layers[2], stride=2)
        self.layer4 = self._make_downlayer(downblock, 512, num_layers[3], stride=2)

        self.uplayer1 = self._make_up_block(upblock, 512, 1, stride=2)
        self.uplayer2 = self._make_up_block(upblock, 256, num_layers[2], stride=2)
        self.uplayer3 = self._make_up_block(upblock, 128, num_layers[1], stride=2)
        self.uplayer4 = self._make_up_block(upblock, 64, 2, stride=2)

        upsample = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels,  # 256
                               64,
                               kernel_size=1, stride=2,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(64),
        )
        self.uplayer_top = DeconvBottleneck(self.in_channels, 64, 1, 2, upsample)

        self.conv1_1 = nn.ConvTranspose2d(64, n_classes, kernel_size=1, stride=1,
                                 bias=False)

    def _make_downlayer(self, block, init_channels, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, init_channels*block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(init_channels*block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, init_channels, stride, downsample))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))

        return nn.Sequential(*layers)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        # expansion = block.expansion
        if stride != 1 or self.in_channels != init_channels * 2:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, init_channels*2,
                                   kernel_size=1, stride=stride,
                                   bias=False, output_padding=1),
                nn.BatchNorm2d(init_channels*2),
            )
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels, 4))
        layers.append(block(self.in_channels, init_channels, 2, stride, upsample))
        self.in_channels = init_channels * 2
        return nn.Sequential(*layers)

    def forward(self, x):
        img = x
        x_size = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        x = self.uplayer_top(x)

        x = self.conv1_1(x, output_size=img.size())

        return x

def copy_parameter_from_resnet50(model, res50_dict):
    cur_state_dict = model.state_dict()
    for name, param in list(res50_dict.items())[0:None]:
        if name not in cur_state_dict: 
            #print('unexpected ', name, ' !')
            continue 
        if isinstance(param, Parameter): 
            param = param.data
        try:
            cur_state_dict[name].copy_(param)
            print('copied ', name)
        except:
            print(name, ' is inconsistent!')
            continue
    print('copy state dict finished!')

@register_model('resnet18')
def resnet18(pretrained=True, classes = 2,**kwargs):
    model = ResNet(Bottleneck, DeconvBottleneck, [2, 2, 2, 2], classes)
    copy_parameter_from_resnet50(model, torchvision.models.resnet34(pretrained = True).state_dict())
    return model


@register_model('resnet34')
def resnet34(pretrained=True, classes=2,**kwargs):
    model = ResNet(Bottleneck, DeconvBottleneck, [3, 4, 6, 3], classes)
    copy_parameter_from_resnet50(model, torchvision.models.resnet34(pretrained = True).state_dict())
    return model

@register_model('resnet50')
def resnet50(pretrained=True, classes=2, **kwargs):
    model = ResNet(Bottleneck, DeconvBottleneck, [3, 4, 6, 3], classes)
    copy_parameter_from_resnet50(model, torchvision.models.resnet50(pretrained = True).state_dict())
    return model

@register_model('resnet101')
def resnet101(pretrained=True, classes=2, **kwargs):
    model = ResNet(Bottleneck, DeconvBottleneck, [3, 4, 23, 3], classes)
    copy_parameter_from_resnet50(model, torchvision.models.resnet101(pretrained = True).state_dict())
    return model

def load_Res50Model(classes):
    model = ResNet(Bottleneck, DeconvBottleneck, [3, 4, 6, 3], classes)
    copy_parameter_from_resnet50(model, torchvision.models.resnet50(pretrained = True).state_dict())
    return model

if __name__ == '__main__':
    net = resnet50()
    img = torch.randn((4,3,256,256))
    print(img.size())
    preds = net(img)
    print(preds.size())