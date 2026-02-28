import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        # 编码器部分
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # #todo: 缺少激活函数模型表达能力受限，缺少归一化，可能产生梯度消失或爆炸
        # self.enc1 = nn.Sequential(
        #     nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True)
        # )
        # self.enc2 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True)
        # )

        self.pool = nn.MaxPool2d(2, 2)

        # 解码器部分
        self.dec1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        # 缺失的激活函数和批归一化层

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        p = self.pool(e2)

        # 解码器
        d1 = self.dec1(p)
        ##todo: 错误的拼接维度
        # 将上采样后的特征图d1与编码器的特征图e1进行拼接。这里假设d1和e1
        # 的空间尺寸相同。然而，由于编码器部分经过两次卷积和一次池化，可能
        # 导致尺寸不匹配，特别是当输入尺寸不是2的幂次方时。
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)

        # print(d1.shape, e1.shape)
        d1 = torch.cat([d1, e1], dim=1)
        d2 = self.dec2(d1)

        out = self.final(d2)
        return out

def compute_flopsNparams1(model):
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)

def compute_flopsNparams2(model):
    import torch
    from thop import profile
    input = torch.randn(1,1,3,256,256)
    flops, params = profile(model, (input))
    print('flops:',flops/1e9,'G', 'params',params/1e6,'M')

def compute_flopsNparams3(model):
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    tensor = (torch.rand(1,3,256,256),)
    flops  = FlopCountAnalysis(model, tensor)
    print("FLOPs:",flops.total()/1e9, 'G')
    print(parameter_count_table(model))
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

if __name__ == '__main__':

    # 实例化模型
    model = UNet()



    # 创建一个示例输入（Batch Size: 1, Channels: 3, Height: 256, Width: 256）
    # input_tensor = torch.randn(1, 3, 256, 256)

    # 前向传播
    # output = model(input_tensor)
    # print(f"输出尺寸: {output.shape}")

    # 使用torchsummary来检查是模型的层次结构和参数信息
    # from torchsummary import summary
    # summary(model, input_size=(3, 224, 224), device='cpu')

    compute_flopsNparams1(model)
    # print("===================")
    # compute_flopsNparams2(model)
    # print("===================")
    # compute_flopsNparams3(model)
    # print("===================")