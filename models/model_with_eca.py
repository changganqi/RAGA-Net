from typing_extensions import Self
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# from pdb import set_trace as stx
from .ScConv_block import ECAResidualBlock, CA_layer
# from utils.antialias import Downsample as downsamp
import cv2
from einops import rearrange

#自己写的GACA+DWT

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ DWT MODULE (No changes needed, this is correct) ++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DWT2d(nn.Module):
    """
    2D Discrete Wavelet Transform using Haar Wavelet.
    This implementation uses fixed convolution filters to decompose the input.
    """

    def __init__(self, in_channels):
        super(DWT2d, self).__init__()
        self.in_channels = in_channels
        ll_filter = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh_filter = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl_filter = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh_filter = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        filters = torch.stack([ll_filter, lh_filter, hl_filter, hh_filter], dim=0).unsqueeze(1)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * 4,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
            groups=in_channels
        )
        self.conv.weight.data = filters.repeat(self.in_channels, 1, 1, 1)
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)


class WaveAttentionCorrelation(nn.Module):
    """
    轻量化 Wave Attention：
    - Q: 原分辨率 1x1 Conv 做瓶颈 (C -> C/r)
    - K/V: DWT 后分组 1x1 Conv（groups=4，对应 4 个子带），输出 2*(C/r)
    - 注意力在窗口内做相关性（无 softmax）
    - 输出用 1x1 Conv 还原通道 (C/r -> C)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, window_size=8, reduction=2):
        super().__init__()
        assert dim % reduction == 0, "dim must be divisible by reduction"
        self.dim = dim
        self.reduction = reduction
        self.dim_q = dim // reduction

        # 头数基于降维后的通道
        assert self.dim_q % num_heads == 0, f"dim_q {self.dim_q} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = self.dim_q // num_heads
        self.scale = self.head_dim ** -0.5

        self.window_size = window_size
        self.dwt_window_size = max(1, window_size // 2)

        self.dwt = DWT2d(in_channels=dim)

        # 1x1 投影：Q 用普通 1x1，K/V 用 groups=4 的 1x1（每个子带一组）
        self.q_proj = nn.Conv2d(dim, self.dim_q, kernel_size=1, bias=qkv_bias)
        self.kv_proj = nn.Conv2d(dim * 4, self.dim_q * 2, kernel_size=1, groups=4, bias=qkv_bias)

        self.proj_out = nn.Conv2d(self.dim_q, dim, kernel_size=1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        ws, dws = self.window_size, self.dwt_window_size

        # --- 新增：自动填充 ---
        # 计算需要填充的量
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        
        # 如果需要，对输入x进行反射填充
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
            # 更新填充后的H和W
            H, W = x.shape[-2:]
        # --- 填充结束 ---

        # 维度检查（可改为自动 pad）
        # assert H % ws == 0 and W % ws == 0, f"H,W must be multiples of window_size={ws}, got {H,W}"
        # assert (H // 2) % dws == 0 and (W // 2) % dws == 0, f"H/2,W/2 must be multiples of dwt_window_size={dws}"

        # Q: (B, Cq, H, W) -> (B, heads, N, Dh)
        q_feat = self.q_proj(x)
        q = rearrange(q_feat, 'b (h d) hh ww -> b h (hh ww) d', h=self.num_heads)

        # 分窗口
        q = rearrange(q, 'b h (rh cw) d -> b h rh cw d', rh=H, cw=W)
        q_windows = rearrange(q, 'b h (rh ws1) (cw ws2) d -> (b rh cw) h (ws1 ws2) d', ws1=ws, ws2=ws)

        # K,V 从 DWT
        x_dwt = self.dwt(x)          # (B, 4C, H/2, W/2)
        H_dwt, W_dwt = H // 2, W // 2
        kv_feat = self.kv_proj(x_dwt) # (B, 2*Cq, H/2, W/2)
        k_feat, v_feat = kv_feat.chunk(2, dim=1)

        k = rearrange(k_feat, 'b (h d) hh ww -> b h (hh ww) d', h=self.num_heads)
        v = rearrange(v_feat, 'b (h d) hh ww -> b h (hh ww) d', h=self.num_heads)

        k = rearrange(k, 'b h (rh cw) d -> b h rh cw d', rh=H_dwt, cw=W_dwt)
        v = rearrange(v, 'b h (rh cw) d -> b h rh cw d', rh=H_dwt, cw=W_dwt)

        k_windows = rearrange(k, 'b h (rh ws1) (cw ws2) d -> (b rh cw) h (ws1 ws2) d', ws1=dws, ws2=dws)
        v_windows = rearrange(v, 'b h (rh ws1) (cw ws2) d -> (b rh cw) h (ws1 ws2) d', ws1=dws, ws2=dws)

        # 相关性注意力（无 softmax）
        corr = (q_windows @ k_windows.transpose(-2, -1)) * self.scale    # (nw, h, ws^2, dws^2)
        attn_windows = corr @ v_windows                                   # (nw, h, ws^2, Dh)

        # 合并窗口 -> (B, Cq, H, W)
        attn_output = rearrange(attn_windows,
                                '(b rh cw) h (ws1 ws2) d -> b (h d) (rh ws1) (cw ws2)',
                                rh=H // ws, cw=W // ws, ws1=ws, ws2=ws)


        # 还原通道
        output = self.proj_out(attn_output)

        # --- 新增：如果进行了填充，裁剪回原始尺寸 ---
        if pad_h > 0 or pad_w > 0:
            # 从填充后的结果中裁剪出有效区域
            output = output[:, :, :H-pad_h, :W-pad_w]
        # --- 裁剪结束 ---

        return output

class FE(nn.Module):
    def __init__(self,
                 in_channels, out_channels1, out_channels2, out_channels3, out_channels4,
                 ksize=3, stride=1, pad=1):
        super(FE, self).__init__()
        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1, 1, 1, 0),
            nn.ReLU())

        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels2, ksize, stride, pad),
            nn.ReLU(),
            nn.Conv2d(out_channels2, out_channels2, ksize, stride, pad),
            nn.ReLU(),
            nn.Conv2d(out_channels2, out_channels2, ksize, stride, pad),
            nn.ReLU())

        self.body3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels3, ksize, stride, pad),
            nn.ReLU(),
            nn.Conv2d(out_channels3, out_channels3, ksize, stride, pad),
            nn.ReLU()
        )

        self.body4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels4, ksize, stride, pad),
            nn.ReLU(),
        )

    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        out3 = self.body3(x)
        out4 = self.body4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class GradiNet_2(nn.Module):
    def __init__(self):
        super(GradiNet_2, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.ELU()
        )
        #
        self.downsample = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=32//12,stride=32//12),
            nn.ELU()
        )
        #
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU()
        )
        #
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=32 // 12, stride=32 // 12),
            nn.ELU()
        )
        #
        self.conv = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
        )
    def forward(self,x):
        x = self.conv0(x)
        for j in range(5):
            resx = x
            x = F.elu(self.res_conv1(x) + resx)

        sp = self.conv(x)

        return sp


class ERFS(nn.Module):
    """
    Event-based Regional Feature Selection (ERFS)
    处理低SNR区域，利用梯度信息引导特征增强
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 梯度提取网络 (from GACA)
        self.fe = FE(3, 32, 32, 32, 32)  # 输入3通道图像，输出128通道特征
        self.g_net = GradiNet_2()  # 输入128通道特征，输出1通道梯度图

        # 特征增强网络
        self.enhancer = nn.Sequential(
            ECAResidualBlock(in_channels),
            ECAResidualBlock(in_channels)
        )
        self.fusion = nn.Conv2d(in_channels + 1, out_channels, 1, 1, 0)  # 融合特征和梯度图

    def forward(self, feature, original_input):
        # 1. 从原始输入图像中提取梯度图
        # --- MODIFICATION: Removed the complex 'with' statement ---
        # The gradient flow is now controlled by the separate optimizers.
        gradient_feature = self.fe(original_input)
        gradient_map = self.g_net(gradient_feature)

        normalized_gradient_map = torch.sigmoid(gradient_map)  # 归一化到0-1，用于特征融合

        # 2. 增强输入特征
        enhanced_feature = self.enhancer(feature)

        # 3. 融合特征和梯度图
        fused = self.fusion(torch.cat([enhanced_feature, normalized_gradient_map], dim=1))
        
        # 返回融合后的特征和未经 sigmoid 的原始梯度图用于计算损失
        return fused, gradient_map

    # <--- NEW: Method to load pre-trained GACA weights --->
    def load_gaca_weights(self, pth_path):
        """Loads weights from a GACA pre-trained model .pth file."""
        print(f"Attempting to load GACA weights for ERFS from: {pth_path}")
        if not os.path.exists(pth_path):
            print(f"Warning: GACA weights file not found at {pth_path}. Skipping.")
            return

        gaca_state_dict = torch.load(pth_path, map_location='cpu')

        # Handle cases where the model is saved inside a dictionary (e.g., 'model_state_dict')
        if 'model' in gaca_state_dict:
            gaca_state_dict = gaca_state_dict['model']
        elif 'state_dict' in gaca_state_dict:
            gaca_state_dict = gaca_state_dict['state_dict']

        # Handle DataParallel prefix 'module.'
        if list(gaca_state_dict.keys())[0].startswith('module.'):
            gaca_state_dict = {k.replace('module.', ''): v for k, v in gaca_state_dict.items()}

        # Extract, rename, and load weights for 'fe' and 'g_net'
        fe_state_dict = {k.replace('fe.', ''): v for k, v in gaca_state_dict.items() if k.startswith('fe.')}
        g_net_state_dict = {k.replace('g_net.', ''): v for k, v in gaca_state_dict.items() if k.startswith('g_net.')}

        if fe_state_dict:
            self.fe.load_state_dict(fe_state_dict)
            print("  - Successfully loaded 'fe' weights.")
        else:
            print("  - Warning: No 'fe' weights found in the GACA pth file.")

        if g_net_state_dict:
            self.g_net.load_state_dict(g_net_state_dict)
            print("  - Successfully loaded 'g_net' weights.")
        else:
            print("  - Warning: No 'g_net' weights found in the GACA pth file.")

    # <--- NEW: Method to freeze/unfreeze GACA module weights --->
    def set_gaca_grad(self, requires_grad=False):
        """Sets the requires_grad attribute for parameters of GACA modules."""
        for param in self.fe.parameters():
            param.requires_grad = requires_grad
        for param in self.g_net.parameters():
            param.requires_grad = requires_grad
        status = "UNFROZEN (trainable)" if requires_grad else "FROZEN (not trainable)"
        print(f"  - GACA modules (fe, g_net) in ERFS set to: {status}")


class SNR_enhance(nn.Module):
    def __init__(self, channel, snr_threshold=0.5, depth=2):
        super().__init__()
        self.channel = channel
        self.depth = depth
        # IRFS: 用于处理高SNR区域的模块
        self.irfs_extractor = nn.ModuleList()
        for i in range(self.depth):
            self.irfs_extractor.append(ECAResidualBlock(self.channel))

        # ERFS: 用于处理低SNR区域的模块
        self.erfs_module = ERFS(in_channels=self.channel, out_channels=self.channel)

        # 融合模块，模仿Trans.py的结构
        self.fea_align = nn.Sequential(
            CA_layer(self.channel * 2),  # 融合来自IRFS和ERFS的特征
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(self.channel * 2, self.channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.threshold = snr_threshold

    def forward(self, cnn_fea, snr_map, original_input):
        """
        cnn_fea: [b,c,h,w] 来自编码器的特征
        snr_map: [b,1,h,w] 对应尺度的SNR图
        original_input: [b,3,h,w] 对应尺度的原始输入图像，用于ERFS
        """
        # 1. 创建高/低SNR区域的掩码
        high_snr_mask = (snr_map > self.threshold).float()
        low_snr_mask = 1.0 - high_snr_mask

        # 2. IRFS分支 (处理高SNR区域)
        irfs_fea = cnn_fea
        for i in range(self.depth):
            irfs_fea = self.irfs_extractor[i](irfs_fea)

        # 3. ERFS分支 (处理低SNR区域)
        erfs_fea, gradient_map = self.erfs_module(cnn_fea, original_input)

        # 4. 根据SNR掩码选择性融合
        out_img = torch.mul(irfs_fea, high_snr_mask)
        out_ev = torch.mul(erfs_fea, low_snr_mask)

        # 5. 特征融合
        # 将两个分支的结果简单相加，然后通过对齐模块
        fused_fea = self.fea_align(torch.cat((out_img, out_ev), dim=1))

        return fused_fea, gradient_map


class ECA(nn.Module):
    """Constructs a ECA module.
    
    Args:
        channels: Number of channels in the input tensor
        b: Hyper-parameter for adaptive kernel size formulation. Default: 1
        gamma: Hyper-parameter for adaptive kernel size formulation. Default: 2 
    """
    def __init__(self, channels, b=1, gamma=2):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size(), padding=(self.kernel_size() - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()


    def kernel_size(self):
        k = int(abs((math.log2(self.channels)/self.gamma)+ self.b/self.gamma))
        out = k if k % 2 else k+1
        return out


    def forward(self, x):
        # [修改] 移除了相位提取 x1=inv_mag(x)
        # feature descriptor on the global spatial information
        # 直接使用输入 x 进行池化，这是传统 ECA 的做法
        y = self.avg_pool(x)


        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)


        # Multi-scale information fusion
        y = self.sigmoid(y)


        return x * y.expand_as(x)



class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        
        # [修改] 移除了 q = inv_mag(q) 和 k = inv_mag(k)
        # 恢复为标准的 Attention 计算，不使用相位信息

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
	def __init__(self, channels, expansion_factor):
		super(GDFN, self).__init__()

		hidden_channels = int(channels * expansion_factor)
		self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
		self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
							  groups=hidden_channels * 2, bias=False)
		self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

	def forward(self, x):
		x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
		x = self.project_out(F.gelu(x1) * x2)
		return x

class attention_to_1x1(nn.Module):
	def __init__(self, channels):
		super(attention_to_1x1, self).__init__()
		self.conv1 = nn.Conv2d(channels, channels*2, kernel_size=1, bias=False)
		self.conv2 = nn.Conv2d(channels*2, channels, kernel_size=1, bias=False)


	def forward(self,x):
		x=torch.mean(x,-1)
		x=torch.mean(x ,-1)
		x=torch.unsqueeze(x ,-1)
		x=torch.unsqueeze(x ,-1)
		xx = self.conv2(self.conv1(x))    
		b, ch, r, c = x.shape
		# print(ch)
		# exit(0)

		return xx
	

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        
        # [新增] 1. 频域分支初始化 (完全模仿 EAMamba)
        # 动态计算 head 数
        num_wave_heads = max(2, (channels // 2) // 32)
        num_wave_heads = max(2, num_wave_heads)
        
        # 实例化 WaveAttentionCorrelation
        # 注意：Restormer 的 Block 没传 bias 参数，默认设为 False 以保持一致
        self.wave_attn = WaveAttentionCorrelation(dim=channels, num_heads=num_wave_heads, qkv_bias=False)

        # [新增] 2. 可学习融合系数 Alpha
        # 初始值为 3.0，经过 sigmoid 后约为 0.95
        # 意味着初始阶段主要信赖 MDTA (0.95)，少量引入频域信息 (0.05)，让网络自己去学最佳比例
        self.alpha_logit = nn.Parameter(torch.tensor(3.0))

        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # --- Attention 分支 ---
        # 1. 归一化
        x_norm1 = x.reshape(b, c, -1).transpose(-2, -1).contiguous()
        x_norm1 = self.norm1(x_norm1)
        x_norm1 = x_norm1.transpose(-2, -1).contiguous().reshape(b, c, h, w)
        
        # 2. 空域分支 (MDTA) - 对应 EAMamba 中的 m_out
        spatial_out = self.attn(x_norm1)
        
        # 3. [新增] 频域分支 (Wave) - 对应 EAMamba 中的 w_out
        wave_out = self.wave_attn(x_norm1)

        # 4. [新增] 可学习融合逻辑
        # 计算 alpha (0~1 之间)
        alpha = torch.sigmoid(self.alpha_logit)
        
        # 融合公式：Res = Input + alpha * Spatial + (1 - alpha) * Frequency
        x = x + alpha * spatial_out + (1 - alpha) * wave_out
        
        # --- FFN 分支 (保持不变) ---
        x_norm2 = x.reshape(b, c, -1).transpose(-2, -1).contiguous()
        x_norm2 = self.norm2(x_norm2)
        x_norm2 = x_norm2.transpose(-2, -1).contiguous().reshape(b, c, h, w)
        x = x + self.ffn(x_norm2)
        
        return x

# class TransformerBlock(nn.Module):
# 	def __init__(self, channels, num_heads, expansion_factor):
# 		super(TransformerBlock, self).__init__()

# 		self.norm1 = nn.LayerNorm(channels)
# 		self.attn = MDTA(channels, num_heads)
# 		self.norm2 = nn.LayerNorm(channels)
# 		self.ffn = GDFN(channels, expansion_factor)

# 	# def forward(self, x):
# 	# 	b, c, h, w = x.shape
# 	# 	x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
# 	# 					  .contiguous().reshape(b, c, h, w))
# 	# 	x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
# 	# 					 .contiguous().reshape(b, c, h, w))
# 	# 	return x

# 	def forward(self, x):
# 			b, c, h, w = x.shape
			
# 			# Attention branch
# 			# 拆分 reshape 和 transpose，确保 contiguous
# 			x_norm1 = x.reshape(b, c, -1).transpose(-2, -1).contiguous()
# 			x_norm1 = self.norm1(x_norm1)
# 			x_norm1 = x_norm1.transpose(-2, -1).contiguous().reshape(b, c, h, w)
# 			x = x + self.attn(x_norm1)
			
# 			# FFN branch
# 			x_norm2 = x.reshape(b, c, -1).transpose(-2, -1).contiguous()
# 			x_norm2 = self.norm2(x_norm2)
# 			x_norm2 = x_norm2.transpose(-2, -1).contiguous().reshape(b, c, h, w)
# 			x = x + self.ffn(x_norm2)
			
# 			return x
			
class DownSample(nn.Module):
	def __init__(self, channels):
		super(DownSample, self).__init__()
		self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
								  nn.PixelUnshuffle(2))

	def forward(self, x):
		return self.body(x)


class UpSample(nn.Module):
	def __init__(self, channels):
		super(UpSample, self).__init__()
		self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
								  nn.PixelShuffle(2))

	def forward(self, x):
		return self.body(x)
  
class UpSample1(nn.Module):
	def __init__(self, channels):
		super(UpSample1, self).__init__()
		self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
								  nn.PixelShuffle(2))

	def forward(self, x):
		return self.body(x)  


class Restormer(nn.Module):
    def __init__(self, num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[16, 32, 64, 128], num_refinement=4,
                 expansion_factor=2.66, ch=[16,16,32,64],
                 # [新增] GACA 参数
                 snr_depth_list=[2, 2, 2],
                 snr_threshold_list=[0.5, 0.5, 0.5],
                 gaca_pth_path=None):
        super(Restormer, self).__init__()
        # self.sig=nn.Sigmoid()
        #self.attention = nn.ModuleList([ECA(num_ch) for num_ch in ch])
       
        self.embed_conv_rgb = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        self.ups1 = UpSample1(32)

        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(8, 3, kernel_size=3, padding=1, bias=False)
        self.output1= nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False)
                                 
        self.ups2 = UpSample1(16)
        self.outputl=nn.Conv2d(32, 8, kernel_size=3, padding=1, bias=False)

        # [新增] GACA 模块初始化
        self.snr_downsample = nn.AvgPool2d(kernel_size=2)
        self.input_downsample = nn.AvgPool2d(kernel_size=2)

        # 1. Bottleneck SNR (对应 channels[3]=128, Scale 1/8)
        self.bottleneck_SNR = SNR_enhance(channels[3], snr_threshold_list[0], snr_depth_list[0])
        self.fusion_bottleneck = nn.Conv2d(channels[3] * 2, channels[3], kernel_size=1)

        # 2. Decoder Level 1 SNR (对应 channels[1]=32, Scale 1/2)
        # 对应 out_dec2 的位置
        self.decoder_level1_SNR = SNR_enhance(channels[1], snr_threshold_list[1], snr_depth_list[1])

        # 3. Decoder Level 2 SNR (对应 channels[1]=32, Scale 1)
        # 对应 fd 的位置 (Refinement 之前)
        self.decoder_level2_SNR = SNR_enhance(channels[1], snr_threshold_list[2], snr_depth_list[2])

        self.snr_modules = nn.ModuleList([
            self.bottleneck_SNR, self.decoder_level1_SNR, self.decoder_level2_SNR
        ])

        if gaca_pth_path:
            self.load_gaca_weights(gaca_pth_path)

    # [新增] 加载 GACA 权重的方法
    def load_gaca_weights(self, pth_path):
        print("--- Loading GACA pre-trained weights for all ERFS modules ---")
        for snr_module in self.snr_modules:
            snr_module.erfs_module.load_gaca_weights(pth_path)
        print("--- Finished loading GACA weights. ---")

    # [新增] 冻结/解冻 GACA 权重的方法
    def set_gaca_grad(self, requires_grad):
        print(f"--- Setting requires_grad={requires_grad} for all GACA modules ---")
        for snr_module in self.snr_modules:
            snr_module.erfs_module.set_gaca_grad(requires_grad)
        print(f"--- Finished setting requires_grad={requires_grad}. ---")
                                 
    def forward(self, RGB_input, snr_map=None):
        # [新增] 处理 snr_map
        if snr_map is None:
            # 如果未提供 snr_map，创建一个默认的 (全0.5，假设中等信噪比)
            snr_map = torch.full((RGB_input.shape[0], 1, RGB_input.shape[2], RGB_input.shape[3]), 0.5, device=RGB_input.device)

        # [新增] 准备多尺度输入和 SNR 图
        # Scale 1
        input_s1 = RGB_input
        snr_s1 = snr_map

        # Scale 1/2
        input_s2 = self.input_downsample(input_s1)
        snr_s2 = self.snr_downsample(snr_s1)

        # Scale 1/4
        input_s4 = self.input_downsample(input_s2)
        snr_s4 = self.snr_downsample(snr_s2)

        # Scale 1/8 (Bottleneck)
        input_s8 = self.input_downsample(input_s4)
        snr_s8 = self.snr_downsample(snr_s4)

        gaca_gradient_maps = []

        # --- Encoder ---
        fo_rgb = self.embed_conv_rgb(RGB_input)
        out_enc_rgb1 = self.encoders[0](fo_rgb)
        out_enc_rgb2 = self.encoders[1](self.downs[0](out_enc_rgb1))
        out_enc_rgb3 = self.encoders[2](self.downs[1](out_enc_rgb2))
        out_enc_rgb4 = self.encoders[3](self.downs[2](out_enc_rgb3))
        
        # [新增] Bottleneck SNR 增强 (Scale 1/8)
        selected_latent, grad_map_b = self.bottleneck_SNR(out_enc_rgb4, snr_s8, input_s8)
        gaca_gradient_maps.append(grad_map_b)
        out_enc_rgb4 = self.fusion_bottleneck(torch.cat([out_enc_rgb4, selected_latent], dim=1))

        # 1. 第一层解码 (去掉 self.attention[0])
        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc_rgb4), out_enc_rgb3], dim=1)))

        # 2. 第二层解码 (去掉 self.attention[1])
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc_rgb2], dim=1)))

        # [中间原本的 GACA 增强逻辑保持不变]
        out_dec2, grad_map_d1 = self.decoder_level1_SNR(out_dec2, snr_s2, input_s2)
        gaca_gradient_maps.append(grad_map_d1)

        # 3. 第三层解码 (去掉 self.attention[2])
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc_rgb1], dim=1))
        
        # [新增] Decoder Level 2 SNR 增强 (Scale 1)
        fd, grad_map_d2 = self.decoder_level2_SNR(fd, snr_s1, input_s1)
        gaca_gradient_maps.append(grad_map_d2)

        fr = self.refinement(fd)  
        outi=self.ups1(fr)
		#out, _ = model(hazy)
		# 返回 (主输出, 梯度图列表)
        return self.output(self.outputl(fr)), gaca_gradient_maps