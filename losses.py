import torch.nn as nn
import torch
from typing import Callable
from kornia.color import rgb_to_grayscale
from kornia.filters import canny, laplacian, sobel
from torch.nn.functional import l1_loss, mse_loss
from torchvision import models
from torchvision.models import vgg16
from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()

        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []

        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i + 1) in indices:
                out.append(X)

        return out

class VGGLoss1(nn.Module):
    def __init__(self, device, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss1, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8]
        self.indices = indices or [2, 7]
        self.device = device
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(self.device)
        else:
            self.normalize = None
        print("Vgg: Weights: ", self.weights, " indices: ", self.indices, " normalize: ", self.normalize)
    def __call__(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

class VGGLoss1(nn.Module):
    def __init__(self, device, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss1, self).__init__()
        if vgg is None:
            self.vgg = Vgg19(requires_grad=False).cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8]
        self.indices = indices or [2, 7]
        self.device = device
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(self.device)
        else:
            self.normalize = None
        print("Vgg: Weights: ", self.weights, " indices: ", self.indices, " normalize: ", self.normalize)
    def __call__(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss

class CharbonnierLoss_MIR(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss_MIR, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    """
    PyTorch module for Edge loss.
    """
    def __init__(self, operator: str = 'canny',
                 loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = l1_loss):
        super(EdgeLoss, self).__init__()
        assert operator in {'canny', 'laplacian', 'sobel'}, 'operator must be one of {canny, laplacian, sobel}'
        self._loss_function = loss_function
        self._operator = operator

    def extract_edges(self, input_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            input_grayscale = rgb_to_grayscale(input_tensor)

            if self._operator == 'canny':
                return canny(input_grayscale)[0]
            elif self._operator == 'laplacian':
                kernel_size = input_grayscale.size()[-1] // 10
                if kernel_size % 2 == 0:
                    kernel_size += 1
                return laplacian(input_grayscale, kernel_size=kernel_size)
            elif self._operator == 'sobel':
                return sobel(input_grayscale)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prediction_edges = self.extract_edges(prediction)
            target_edges = self.extract_edges(target)

            return self._loss_function(prediction_edges, target_edges)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight


    def forward(self, x):
        x = x.unsqueeze(0)
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])  # 算出总共求了多少次差
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        # x[:,:,1:,:]-x[:,:,:h_x-1,:]就是对原图进行错位，分成两张像素位置差1的图片，第一张图片
        # 从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个
        # 像素点，这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相
        # 邻的下一个像素点的差。
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class CharbonnierLoss(nn.Module):
    def __init__(self,epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon2=epsilon*epsilon

    def forward(self,x):
        value=torch.sqrt(torch.pow(x,2)+self.epsilon2)
        return torch.mean(value)

class ContentLoss(nn.Module):
    def __init__(self, weights=[1.0,1.0,1.0,1.0,1.0]):
        super().__init__()
        features = models.vgg16(pretrained=True).cuda().features
        self.loss = torch.nn.L1Loss().cuda()
        self.weights = weights
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):

        total_loss = 0
        for block, weight in zip([self.to_relu_1_2, self.to_relu_2_2, self.to_relu_3_3, self.to_relu_4_3], self.weights):
            x, y = block(x), block(y)
            total_loss += weight*self.loss(x, y)
        return total_loss/5.0

# --- Perceptual loss network  --- #
class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.cuda()
        for param in vgg_model.parameters():
            param.requires_grad = False

        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(mse_loss(dehaze_feature, gt_feature))

        return sum(loss)/len(loss)
if __name__ == '__main__':
    x = torch.rand(1,3,256,256)
    y = x-5
    closs = ContentLoss()
    print(closs(x,y))


