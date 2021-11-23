from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
import torch

__all__ = ['sdc_resnet32_norm']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class resnet_cifar(nn.Module):

    def __init__(self, block, num_blocks,  pretrained=False, cut_at_pooling=False,
                 Embed_dim=512, norm=True, dropout=0, num_classes=0):
        super(resnet_cifar, self).__init__()
        self.Embed_dim = Embed_dim  #lu adds
        self.out_dim = Embed_dim
        self.num_features = Embed_dim
        self.pretrained = pretrained
        print(50*"*" + "pretrained")
        print(pretrained)
        self.in_planes = 16
        self.cut_at_pooling = cut_at_pooling
 
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.base_feature = nn.Sequential(self.conv1, self.bn1, self.relu, self.layer1, self.layer2, self.layer3)
        self.base = nn.Sequential()
        self.base.fc = nn.Linear(64, 1000)
        

        if not self.cut_at_pooling:
            self.num_features = Embed_dim
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = Embed_dim > 0
            self.num_classes = num_classes

            # Append new layers
            if self.has_embedding:
                self.base.fc = nn.Linear(64, self.num_features)
                init.kaiming_normal_(self.base.fc.weight, mode='fan_out')
                init.constant_(self.base.fc.bias, 0)
               
            self.num_features = 64
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)
        
        #self.Embed is not used. why it is defined?
        self.Embed = Embedding(64, 1)
        if self.Embed_dim == 0: 
            pass
        else:
            self.Embed = Embedding(64, 1)

        if not self.pretrained:
            self.reset_params()
        else:
            model_dict = self.state_dict()
            pretrained_dict = torch.load(
                            '/data/code/continual-learning-reproduce/pretrained_models/Finetuning_0_task_0_200_model_task2_cifar100_seed1993.pt')
            print("load pretrained model")
            # pretrained_dict = pretrained_dict.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items(
            ) if k in model_dict and 'fc' not in k}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        
        x = self.base_feature(x)
        if self.cut_at_pooling:
            return x
        
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x_feat = x

        if self.has_embedding:
            x = self.base.fc(x)
        if self.norm:
            # print("normalize")
            x = F.normalize(x)
        elif self.has_embedding:
            # print("has_embedding")
            x = F.relu(x)
        if self.dropout > 0:
            # print("dropout")
            x = self.drop(x)
        if self.num_classes > 0:
            # print("num_classes")
            x = F.relu(x)
            x = self.classifier(x)
        return {"features": x}
    
    def inference(self, x):
        x = self.base_feature(x)
        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.base.fc(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        
        if self.num_classes > 0:
            x = F.relu(x)

        return x 

    def forward_without_norm(self, x):
        x = self.base_feature(x)
        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.base.fc(x)
        if self.num_classes > 0:
            x = F.relu(x)
            x = self.classifier(x)
        
        return {"features": x}
    
    def extract_feat(self, x):
        x = self.base_feature(x)
        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class Embedding(nn.Module): 
    def __init__(self, in_dim, out_dim, dropout=None, normalized=True):
        super(Embedding, self).__init__()
        self.bn = nn.BatchNorm2d(in_dim, eps=0.001)
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.dropout = dropout
        self.normalized = normalized

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        if self.dropout is not None:
            x = nn.Dropout(p=self.dropout)(x, inplace=True)
        x = self.linear(x)
        if self.normalized:
            norm = x.norm(dim=1, p=2, keepdim=True)
            x = x.div(norm.expand_as(x))
        return x

def sdc_resnet32_norm(**kwargs):
    print(kwargs)
    model = resnet_cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model

if __name__ == "__main__":
    params = {}
    model = sdc_resnet32_norm(**params)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(
                    '../pretrained_models/Finetuning_0_task_0_200_model_task2_cifar100_seed1993.pt')
    # pretrained_dict = pretrained_dict.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items(
    ) if k in model_dict and 'fc' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print(model)