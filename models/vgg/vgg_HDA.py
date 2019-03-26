import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'model'
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, args=None):
        super(VGG, self).__init__()
        self.conv1_2 = nn.Sequential(*features[:10])
        self.conv3 = nn.Sequential(*features[10:17])
        self.conv4 = nn.Sequential(*features[17:24])
        # self.conv1_4 = nn.Sequential(*features[:-5])
        self.conv5 = nn.Sequential(*features[24:-1])
        self.fmp = features[-1]  # final max pooling

        # added layer
        self.fc3_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
        )
        self.fc3_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),  # fc7
            nn.ReLU(True),
        )
        self.cls3 = nn.Conv2d(512, 11, kernel_size=1, padding=0)  #

        self.fc4_1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
        )
        self.fc4_2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
            nn.ReLU(True),
        )
        self.cls4 = nn.Conv2d(1024, 37, kernel_size=1, padding=0)  #

        self.fc5_1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
        )
        self.fc5_2 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
            nn.ReLU(True),
        )
        self.cls5 = nn.Conv2d(1024, num_classes, kernel_size=1, padding=0)  #


        self._initialize_weights()

        # loss function
        self.loss_cross_entropy = nn.CrossEntropyLoss()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        # ======================================================================

        x = self.conv1_2(x)
        x = self.conv3(x)

        rootResult = self.fc3_1(x)
        rootResult = self.fc3_2(rootResult)
        rootResult = self.cls3(rootResult)
        root_logits = torch.mean(torch.mean(rootResult, dim=2), dim=2)


        x = self.conv4(x)

        parentResult = self.fc4_1(x)
        parentResult = self.fc4_2(parentResult)
        parentResult = self.cls4(parentResult)
        parent_logits = torch.mean(torch.mean(parentResult, dim=2), dim=2)
        # # ======================================================================
        # =============================== Result4 ==============================

        x = self.conv5(x)

        childResult = self.fc5_1(x)
        childResult = self.fc5_2(childResult)
        childResult = self.cls5(childResult)
        child_logits = torch.mean(torch.mean(childResult, dim=2), dim=2)
        # ======================================================================


        self.child_map = childResult
        self.parent_map = parentResult
        self.root_map = rootResult

        return root_logits, parent_logits, child_logits

    def get_loss(self, logits, gt_root_label, gt_parent_label, gt_child_label):
        root_logits, parent_logits, child_logits = logits

        # ======================================================================
        #
        root_loss_cls = self.loss_cross_entropy(root_logits, gt_root_label.long())
        parent_loss_cls = self.loss_cross_entropy(parent_logits, gt_parent_label.long())
        child_loss_cls = self.loss_cross_entropy(child_logits, gt_child_label.long())


        loss_val = 0.5 * root_loss_cls + 0.5 * parent_loss_cls + 0.5 * child_loss_cls
        return loss_val, root_loss_cls, parent_loss_cls, child_loss_cls

    def get_localization_maps(self):
        return self.normalize_atten_maps(self.cam)

    def get_child_maps(self):
        return self.child_map

    def get_parent_maps(self):
        return self.parent_map

    def get_root_maps(self):
        return self.root_map




def make_layers(cfg, dilation=None, batch_norm=False):
    layers = []
    in_channels = 3
    for v, d in zip(cfg, dilation):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'L':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d, dilation=d)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'O': [64, 64, 'L', 128, 128, 'L', 256, 256, 256, 'L', 512, 512, 512, 'L', 512, 512, 512, 'L']
}

dilation = {
    'D1': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'N', 1, 1, 1, 'N']
}


def model(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    """
    model = VGG(make_layers(cfg['O'], dilation=dilation['D1']), **kwargs)
    if pretrained:
        pre2local_keymap = [('features.{}.weight'.format(i), 'conv1_2.{}.weight'.format(i)) for i in range(10)]
        pre2local_keymap += [('features.{}.bias'.format(i), 'conv1_2.{}.bias'.format(i)) for i in range(10)]
        pre2local_keymap += [('features.{}.weight'.format(i + 10), 'conv3.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 10), 'conv3.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.weight'.format(i + 17), 'conv4.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 17), 'conv4.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.weight'.format(i + 24), 'conv5.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 24), 'conv5.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap = dict(pre2local_keymap)

        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
        print('load pretrained model from {}'.format(model_urls['vgg16']))
        # 0. replace the key
        pretrained_dict = {pre2local_keymap[k] if k in pre2local_keymap.keys() else k: v for k, v in
                           pretrained_dict.items()}
        # *. show the loading information
        for k in pretrained_dict.keys():
            if k not in model_dict:
                print('Key {} is removed from vgg16'.format(k))
        print(' ')
        for k in model_dict.keys():
            if k not in pretrained_dict:
                print('Key {} is new added for DA Net'.format(k))
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    model(True)
