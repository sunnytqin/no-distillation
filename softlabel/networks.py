import torch.nn as nn
import torch.nn.functional as F
import torch
# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

# adapted from
# https://github.com/VICO-UoE/DatasetCondensation



''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features_a, self.features_b, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x, feature_only=False):
        out = self.features_a(x)
        if feature_only == 1:
            out = out.view(out.size(0), -1)
            return out
        out = self.features_b(out)
        out = out.view(out.size(0), -1)
        if feature_only == 2:
            return out
        out = self.classifier(out)
        return out

    #     self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
    #     num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
    #     self.classifier = nn.Linear(num_feat, num_classes)

    # def forward(self, x):
    #     # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
    #     out = self.features(x)
    #     out = out.view(out.size(0), -1)
    #     out = self.classifier(out)
    #     return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2


        return nn.Sequential(*layers[0: len(layers)//2+1]), nn.Sequential(*layers[len(layers)//2+1: ]), shape_feat
    
''' ConveNet_MoE '''
class ConvNet_MoE(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32), num_experts=5):
        super(ConvNet_MoE, self).__init__()

        self.num_experts = num_experts
        self.experts = nn.ModuleList([ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size) for _ in range(num_experts)])
        self.gate = ConvNet(channel, num_experts, net_width, 2, net_act, net_norm, net_pooling, im_size)
        # self.gate_noise = ConvNet(channel, num_experts, net_width, net_depth, net_act, net_norm, net_pooling, im_size)
        self.gate_k = 2

    def forward(self, x):
        # compute sparse gate weights
        gate_out = self.gate(x)
        # gate_noise_weight = self.gate_noise(x)
        # gate_out = gate_out + F.softplus(gate_noise_weight) #* torch.randn_like(gate_noise_weight)
        # zero out non top-k
        top_k_values, _ = torch.topk(gate_out, self.gate_k, dim=1)
        gate_out[gate_out < top_k_values[:, -1].unsqueeze(1)] = float('-inf')
        gate_out = F.softmax(gate_out, dim=1)

        # compute expert outputs
        expert_out = [expert(x) for expert in self.experts]
        expert_out = torch.stack(expert_out, dim=1)

        # combine
        out = torch.sum(gate_out.unsqueeze(2) * expert_out, dim=1)
        return out
