import torch
import models.archs.low_light_transformer as low_light_transformer
import torch
import torch.nn as nn
import functools
import models.eamamba as eamamba
# 假设您的残差块等工具函数在这个文件里
# 如果不在，您可能需要从 SNR-Aware-Low-Light-Enhance 项目中复制 models/archs/arch_util.py
try:
    import models.archs.arch_util as arch_util
except ImportError:
    print("警告: 'models.archs.arch_util' 未找到。将使用简化的残差块。")
    # 提供一个备用的简单实现，以防文件不存在
    class ResidualBlock_noBN(nn.Module):
        def __init__(self, nf=64):
            super(ResidualBlock_noBN, self).__init__()
            self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        def forward(self, x):
            identity = x
            out = self.lrelu(self.conv1(x))
            out = self.conv2(out)
            return identity + out
    arch_util = type('module', (), {'ResidualBlock_noBN': ResidualBlock_noBN, 'make_layer': lambda block, n_layers: nn.Sequential(*[block() for _ in range(n_layers)])})


class SNRAware_EAMamba_Hybrid(nn.Module):
    """
    复现您论文的双分支结构，但用 EAMamba 替换 Transformer。
    - 短距离分支: 卷积结构，用于局部细节。
    - 长距离分支: EAMamba，用于全局上下文和光照。
    - SNR感知: forward 方法接收一个 mask 来引导增强过程。
    """
    def __init__(self, nf=64, front_RBs=5, back_RBs=10, short_branch_RBs=5):
        super(SNRAware_EAMamba_Hybrid, self).__init__()
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        #### 1. 共享的特征提取前端
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)

        #### 2. 短距离分支 (Short-Range Branch)
        self.short_range_branch = arch_util.make_layer(ResidualBlock_noBN_f, short_branch_RBs)

        #### 3. 长距离分支 (Long-Range Branch with EAMamba)
        self.long_range_branch = eamamba.EAMamba(
            inp_channels=nf, out_channels=nf, dim=nf,
        num_blocks=[4, 6, 6, 7],
        num_refinement_blocks=2,
        ffn_expansion_factor=2.0,
        bias=False,
        layernorm_type='WithBias',
        dual_pixel_task=False,
        checkpoint_percentage=0.0,
        channel_mixer_type='Simple',
        upscale=1,  # 根据您的示例，这里设置为1
        mamba_cfg={
            'scan_type': 'diagonal',
            'scan_count': 8,
            'scan_merge_method': 'concate',
            'disable_z_branch': False,
            'd_state': 16,
            'd_conv': 3,
            'expand': 1,
            'conv_2d': True
        }
    )

        #### 4. 分支融合
        self.fusion = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)

        #### 5. 共享的重建后端
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, mask):
        """
        接收低光图像 x 和 SNR 感知掩码 mask。
        """
        # 1. 共享特征提取
        fea = self.lrelu(self.conv_first(x))
        fea = self.feature_extraction(fea)

        # 2. 双分支处理
        short_range_out = self.short_range_branch(fea)
        long_range_out = self.long_range_branch(fea)

        # 3. SNR 感知融合
        # 根据论文思想，mask引导高频细节(短距离)和低频光照(长距离)的融合
        # 一个简单的实现是使用 mask 来加权融合
        fused_fea = self.lrelu(self.fusion(torch.cat([
            short_range_out * (1 - mask),  # 低SNR区域更依赖短距离分支(去噪)
            long_range_out * mask          # 高SNR区域更依赖长距离分支(细节)
        ], dim=1)))

        # 4. 重建
        out = self.recon_trunk(fused_fea)
        out = self.conv_last(out)
        out += x # 全局残差连接
        return out
        
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'low_light_transformer':
        netG = low_light_transformer.low_light_transformer(nf=opt_net['nf'], nframes=opt_net['nframes'],
                                                           groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                                           back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                                                           predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                                                           w_TSA=opt_net['w_TSA'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

