import torch
import torch.nn as nn
import torchvision
import os
import argparse
import time
from models.model_with_eca import Restormer
# import models.SFHformer as SFHformer

import numpy as np
from PIL import Image
import glob

parser = argparse.ArgumentParser()
# parser.add_argument('--filepath', type=str, default='/home/s_yzm/cqq/dataset/EUVP-200/raw-200')
# parser.add_argument('--filepath', type=str, default='/home/s_yzm/cqq/dataset/EUVP-200/ref-200')
# parser.add_argument('--hazy_images_path_test', type=str, default=r"/home/s_yzm/cqq/dataset/EUVP-200/raw-200")
# parser.add_argument('--ref_images_path_test', type=str, default=r"/home/s_yzm/cqq/dataset/EUVP-200/ref-200")
# parser.add_argument('--filepath', type=str, default='/home/s_yzm/cqq/dataset/UIEB-100/test/raw')
# parser.add_argument('--snr_filepath', type=str, default='/home/s_yzm/cqq/dataset/UIEB-100/snr-100')

parser.add_argument('--filepath', type=str, default='/home/s_yzm/cqq/dataset/U45/test/raw')
parser.add_argument('--snr_filepath', type=str, default='/home/s_yzm/cqq/dataset/U45/snr-u45/')

parser.add_argument('--name', type=str, default='U45_GAWT')###
parser.add_argument('--pretrain_path', type=str, default='/home/s_yzm/cqq/ba/training_data/UIEBWACVGAWT最终/weight/best_0.9219.pth')
parser.add_argument('--dim_hist', type=int, default=64) #/data1/dataset/underwater/UIEB-100/raw-100
args = parser.parse_args()
 
def lowlight(image_path, snr_path, color_net):
    
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight)/255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)
    
    if snr_path and os.path.exists(snr_path):
        data_snr = Image.open(snr_path)
        data_snr = (np.asarray(data_snr)/255.0)
        data_snr = torch.from_numpy(data_snr).float()
        # 自动处理通道维度
        if len(data_snr.shape) == 2:
            data_snr = data_snr.unsqueeze(0)
        elif len(data_snr.shape) == 3:
            data_snr = data_snr.permute(2, 0, 1)
        data_snr = data_snr.cuda().unsqueeze(0)
    else:
        data_snr = None
        print(f"Warning: SNR map not found for {image_path}, skipping SNR enhancement.")

    # # --- 核心修改：在这里调整尺寸 ---
    # h, w = data_lowlight.shape[2], data_lowlight.shape[3]
    # new_h = (h // 8) * 8
    # new_w = (w // 8) * 8
    # data_lowlight = nn.functional.interpolate(data_lowlight, size=(new_h, new_w), mode='bilinear', align_corners=False)
    # # --- 修改结束 ---

    with torch.no_grad():
        start = time.time()
        # [修改 2] Restormer 返回 (out, out_2x)，需要解包，只取第一个输出
        # 传入 data_snr
        enhanced_image, _ = color_net(data_lowlight, data_snr)
    end_time = (time.time() - start)
    print(end_time)

    result_path = './results/' + args.name + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    torchvision.utils.save_image(enhanced_image, result_path + image_path.split("/")[-1])

if __name__ == '__main__':
    
    with torch.no_grad():
        # model setting
        os.environ['CUDA_VISIBLE_DEVICES']='2'
        #color_net = return_model().cuda()
        color_net = Restormer().cuda() 
        

                    # 处理 nn.DataParallel 包装导致的 'module.' 前缀
        from collections import OrderedDict
        saved_state_dict = torch.load(args.pretrain_path)

        new_state_dict = OrderedDict()
        for k, v in saved_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        color_net.load_state_dict(new_state_dict)
        # path setting
        filePath = args.filepath
        snrPath = args.snr_filepath # 获取 SNR 文件夹路径
        test_list = glob.glob(filePath + "/*")

        # inference
        for image in test_list:
            print(image)
            # --- 构建对应的 SNR 图片路径 ---
            image_name = os.path.basename(image)
            current_snr_path = os.path.join(snrPath, image_name) if snrPath else None
            
            lowlight(image, current_snr_path, color_net)
