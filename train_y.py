import torch.optim
import argparse
import time
import dataloader
from SSIM import SSIM, SSIMLOSS
import torchvision
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import warnings
# [修改 1] 导入新的网络模型
from models.model_with_eca import Restormer

warnings.filterwarnings('ignore')
import datetime
from util import *
from losses import *
import sys
import numpy
import os
import shutil

# [已删除] import models.SFHformer as SFHformer
def get_gt_grad(ref_image):
# ...existing code...
    """
    使用拉普拉斯算子从参考图像生成单通道梯度真值图。
    """
    img_tensor = ref_image.float()
    B, C, H, W = img_tensor.shape
    device, dtype = img_tensor.device, img_tensor.dtype

    # 拉普拉斯核 (中心8，其他-1)，重复到3个通道
    lap_base = torch.tensor([[-1, -1, -1],
                             [-1, 8, -1],
                             [-1, -1, -1]], dtype=dtype, device=device)

    # [C,1,3,3]，每个通道一个相同的核
    lap_kernel = lap_base.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    # 分组卷积，每个通道独立应用
    lap_resp = F.conv2d(img_tensor, lap_kernel, padding=1, groups=C)

    # 取绝对值作为梯度强度
    gradient_lap = lap_resp.abs()

    # 取均值得到单通道“强度图”
    gradient_lap_gray = gradient_lap.mean(dim=1, keepdim=True)

    return gradient_lap_gray



def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.02)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


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


def train(config):
    # writer = SummaryWriter(comment=config.modelname)

    # [修改 2] 实例化 Restormer，传入 GACA 参数
    model = Restormer(
        snr_depth_list=config.snr_depth_list,
        snr_threshold_list=config.snr_threshold_list,
        gaca_pth_path=config.gaca_pth_path
    ).cuda() 


    use_dataparallel = torch.cuda.device_count() > 1
    if use_dataparallel:
        model = nn.DataParallel(model)

    epoch_start = 0

    # [修改 3] 传入 snr_images_path

    train_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path,
                                               snr_images_path=config.snr_images_path, mode="train", resize=(256, 256))
    val_dataset = dataloader.dehazing_loader(config.orig_images_path_val, config.hazy_images_path_val,
                                             snr_images_path=config.snr_images_path_val, mode="val", resize=(256, 256))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,
                                             num_workers=config.num_workers, pin_memory=True)

    criterion = []
    vgg = Vgg19(requires_grad=False).to('cuda:0')
    vgg_loss = VGGLoss1(device='0', vgg=vgg, normalize=False)
    criterion.append(SSIM().cuda())
    criterion.append(nn.MSELoss().cuda())
    criterion.append(vgg_loss.cuda())
    criterion.append(nn.L1Loss().cuda()) # L1 Loss for GACA
    criterion.append(nn.L1Loss().cuda())
    comput_ssim = SSIM()

    # [修改 4] 分离参数，使用两个优化器
    main_params = []
    gaca_params = []
    for name, param in model.named_parameters():
        if 'erfs_module.fe' in name or 'erfs_module.g_net' in name:
            gaca_params.append(param)
        else:
            main_params.append(param)

    optimizer_main = torch.optim.AdamW(main_params, lr=config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)
    optimizer_gaca = torch.optim.AdamW(gaca_params, lr=config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)

    model.train()
    Iters = 1

    indexX = []
    indexY = []

    for epoch in range(epoch_start, epoch_start + config.num_epochs):

        # 手动调整学习率
        if epoch <= 100:
            current_lr = 0.0001
        elif epoch <= 150:
            current_lr = 0.00008
        elif epoch <= 300:
            current_lr = 0.00005
        elif epoch <= 400:
            current_lr = 0.00003
        else:  # epoch <= 500
            current_lr = 0.00001

        # 更新优化器的学习率
        for param_group in optimizer_main.param_groups:
            param_group['lr'] = current_lr
        for param_group in optimizer_gaca.param_groups:
            param_group['lr'] = current_lr

        print(f"Current learning rate == {optimizer_main.param_groups[0]['lr']}")
        print("*" * 60 + 'Epoch: ' + str(epoch - epoch_start) + "*" * 60)

        loop = tqdm(enumerate(train_loader), total=len(train_loader), ncols=120, colour="red")
        # [修改 5] 接收 snr_map
        for iteration, (gt, hazy, snr_map, extension) in loop:
            gt = gt.cuda()
            hazy = hazy.cuda()
            snr_map = snr_map.cuda()

            # 检查 SNR map 是否有效
            if snr_map.dim() == 2 and snr_map[0, 0] == -1.0:
                 raise ValueError(f"训练时未找到SNR图，请检查路径: {config.snr_images_path}")
            elif snr_map.dim() == 4 and snr_map[0, 0, 0, 0] == -1.0:
                raise ValueError(f"训练时未找到SNR图，请检查路径: {config.snr_images_path}")

            try:
                # [修改 6] 前向传播，接收 snr_map，获取梯度图
                out, gaca_grads = model(hazy, snr_map)
                
                # [修改 7] 计算 GACA L1 损失
                loss_gaca_l1 = 0
                current_gt_for_grad = gt
                # gaca_grads 列表顺序是 [grad_b, grad_d1, grad_d2] (从小到大: 1/8, 1/2, 1)
                # 注意：Restormer 中我们添加的顺序是 Bottleneck(1/8), Decoder1(1/2), Decoder2(1)
                # 所以不需要 reversed，直接遍历即可，或者根据尺寸匹配
                for i, pred_grad in enumerate(gaca_grads): 
                    if current_gt_for_grad.shape[2:] != pred_grad.shape[2:]:
                        current_gt_for_grad = F.interpolate(gt, size=pred_grad.shape[2:], mode='bilinear', align_corners=False)
                    else:
                        current_gt_for_grad = gt
                    
                    gt_grad = get_gt_grad(current_gt_for_grad)
                    loss_gaca_l1 += criterion[3](pred_grad, gt_grad)

                ssim_loss = 1 - criterion[0](out, gt)
                
                # 总损失
                loss = ssim_loss + loss_gaca_l1

                optimizer_main.zero_grad()
                optimizer_gaca.zero_grad()
                
                loss.backward()

                # 添加梯度裁剪，防止梯度消失/爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer_main.step()
                optimizer_gaca.step()
                
                Iters += 1

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(e)
                    torch.cuda.empty_cache()
                else:
                    raise e

            loop.set_description(f'Epoch [{epoch}/{config.num_epochs}]')
            loop.set_postfix(Loss=loss.item(), SSIM=ssim_loss.item(), GACA=loss_gaca_l1.item())

        val_ssim = []
        print("start Val!")
        mkdir(config.sample_output_folder + f"/epochs/epoch{str(epoch).zfill(3)}")
        with torch.no_grad():
            loop_val = tqdm(enumerate(val_loader), total=len(val_loader), ncols=100, colour='green')
            # [修改 8] 验证集也接收 snr_map
            for id, (val_y, val_x, val_snr_map, extension) in loop_val:
                val_y = val_y.cuda()
                val_x = val_x.cuda()
                val_snr_map = val_snr_map.cuda()

                if val_snr_map[0, 0, 0, 0] == -1.0:
                    raise ValueError(f"验证时未找到SNR图，请检查路径: {config.snr_images_path_val}")

                # [修改 9] 验证集前向传播，处理元组返回
                model_output = model(val_x, val_snr_map)
                if isinstance(model_output, tuple):
                    val_out = model_output[0]
                else:
                    val_out = model_output
                
                iter_ssim = comput_ssim(val_y, val_out)
                val_ssim.append(iter_ssim.item())

                torchvision.utils.save_image(torch.cat((val_x, val_y, val_out), 0),
                                             f"{config.sample_output_folder}/epochs/epoch{str(epoch).zfill(3)}/{str(id + 1).zfill(3)}{extension[0]}")

                loop_val.set_description(f'VAL')
                loop_val.set_postfix(ssim=iter_ssim.item())

            indexX.append(epoch)
            now = np.mean(val_ssim)
            ssim_log = "[%i,%f]" % (epoch, now) + "\n"

            if indexY == []:
                indexY.append(now)
                print("First epoch，Save！", 'Now Epoch mean SSIM is:', now)
                torch.save(model.state_dict(), config.snapshots_folder + 'first.pth')
                print('saved first pth!')
            else:
                now_max_ssim = max(indexY)
                indexY.append(now)
                print('Best SSIM:', now_max_ssim, 'Now Epoch mean SSIM is:', now)

                if now >= now_max_ssim:
                    weight_name = f"best_{now:.4f}.pth"
                    torch.save(model.state_dict(), config.snapshots_folder + weight_name)
                    print("\033[31msave pth!！！！！！！！！！！！！！！！！！！！！\033[0m")
                    ssim_log = "[%i,%f]*" % (epoch, np.mean(val_ssim)) + "\n"
                if now < now_max_ssim and epoch > 10:  # 从第10轮后开始删除效果不好的模型保存的图片
                    shutil.rmtree(config.sample_output_folder + f"/epochs/epoch{str(epoch).zfill(3)}")

            with open('training_data/%s/train.log' % config.modelname, "a+", encoding="utf-8") as f:
                f.write(ssim_log)


if __name__ == "__main__":
    defaultname = "UIEBWACVGAWT" + datetime.datetime.now().strftime("%m%d-%H%M")
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)  # 初始学习率
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--snapshots_folder', type=str, default="training_data/%s/weight/" % defaultname)
    parser.add_argument('--sample_output_folder', type=str, default="training_data/%s/sample/" % defaultname)
    parser.add_argument('--results_output_folder', type=str, default="training_data/%s/sample/results/" % defaultname)
    parser.add_argument('--modelname', type=str, default=defaultname)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=8)

    # [修改 10] 添加 SNR 相关参数
    parser.add_argument('--snr_depth_list', type=int, nargs='+', default=[2, 2, 2], help='List of depths for SNR_enhance modules.')
    parser.add_argument('--snr_threshold_list', type=float, nargs='+', default=[0.5, 0.5, 0.5], help='List of thresholds for SNR_enhance modules.')
    parser.add_argument('--gaca_pth_path', type=str, default=None, help="Path to pre-trained GACA model weights (.pth file).")
    
    parser.add_argument('--snr_images_path', type=str, default=r"/home/s_yzm/cqq/dataset/UIEB-100/snr-790", help="Path to training SNR maps")
    parser.add_argument('--snr_images_path_val', type=str, default=r"/home/s_yzm/cqq/dataset/UIEB-100/snr-100", help="Path to validation SNR maps")

    #--- 数据集路径 ---
    parser.add_argument('--orig_images_path', type=str, default=r"/home/s_yzm/cqq/dataset/UIEB-100/train/ref")
    parser.add_argument('--hazy_images_path', type=str, default=r"/home/s_yzm/cqq/dataset/UIEB-100/train/raw")
    parser.add_argument('--orig_images_path_val', type=str, default=r"/home/s_yzm/cqq/dataset/UIEB-100/test/ref")
    parser.add_argument('--hazy_images_path_val', type=str, default=r"/home/s_yzm/cqq/dataset/UIEB-100/test/raw")

    # parser.add_argument('--snr_images_path', type=str, default=r"/home/s_yzm/cqq/dataset/EUVP-200/snr-1985", help="Path to training SNR maps")
    # parser.add_argument('--snr_images_path_val', type=str, default=r"/home/s_yzm/cqq/dataset/EUVP-200/snr-200", help="Path to validation SNR maps")


    # parser.add_argument('--orig_images_path', type=str, default=r"/home/s_yzm/cqq/dataset/EUVP-200/train/ref")
    # parser.add_argument('--hazy_images_path', type=str, default=r"/home/s_yzm/cqq/dataset/EUVP-200/train/raw")
    # parser.add_argument('--orig_images_path_val', type=str, default=r"/home/s_yzm/cqq/dataset/EUVP-200/test/ref")
    # parser.add_argument('--hazy_images_path_val', type=str, default=r"/home/s_yzm/cqq/dataset/EUVP-200/test/raw")


    parser.add_argument('--cudaid', type=str, default="1,2", help="choose cuda device id 0-7.")

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cudaid

    mkdir("training_data")

    previous_training = os.listdir("training_data")
    for src_folder in previous_training:
        src_folder = os.path.join("training_data", src_folder)
        train_log_file = os.path.join(src_folder, "train.log")
        if os.path.exists(train_log_file):
            with open(train_log_file, 'r') as file:
                line_count = sum(1 for line in file)
            if line_count < 2:
                try:
                    shutil.rmtree(src_folder)
                    print(f"文件夹 '{src_folder}' 已删除。")
                except OSError as e:
                    print(f"删除文件夹 '{src_folder}' 失败: {e}")
        else:
            pass

    mkdir("training_data/%s" % config.modelname)
    mkdir("training_data/%s/project_files" % (config.modelname))
    mkdir(config.snapshots_folder)
    mkdir(config.sample_output_folder)
    mkdir(config.sample_output_folder + "/best")
    mkdir(config.sample_output_folder + "/epochs")
    mkdir(config.results_output_folder)
    with open("training_data/%s/train.log" % config.modelname, 'w') as file:
        pass
    with open("training_data/%s/记录.txt" % config.modelname, 'w') as file:
        pass

    current_time = datetime.datetime.now()
    with open('training_data/%s/project_files/%s.txt' % (config.modelname, config.modelname), "w") as f:
        for i in vars(config):
            f.write(i + ":" + str(vars(config)[i]) + '\n')
        f.write("train time:%s" % str(current_time))

    copy_project_files(os.getcwd(), 'training_data/%s/project_files' % config.modelname)
    time.sleep(2)

    try:
        train(config)
        sys.exit()
    except Exception as e:
        raise (e)
        torch.cuda.empty_cache()
        time.sleep(2)
        sys.exit()