import torch
import glob
import shutil
import os
import subprocess

def catDissplayImg(t):
    if type(t) == tuple:
        t = [*t]
    if type(t) == list:
        for i in range(len(t)):
            if i == 0:
                dis = t[i]
            else:
                dis = torch.cat((dis, t[i]), 0)
        return dis
    else:
        return t


def mkdir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        print("Create directory: ", directory)
    else:
        print(directory, " already exists.")


def transfPY(config):
    fileList = glob.glob("*.py")
    for fileName in fileList:
        shutil.copy(fileName, 'training_data/%s/project_files/%s' % (config.modelname, fileName))

def copy_project_files(src_dir, dest_dir):
    # 遍历当前目录下的所有文件和文件夹
    for item in os.listdir(src_dir):
        item_path = os.path.join(src_dir, item)

        # 如果是文件夹，则递归调用函数进行复制
        if os.path.isdir(item_path) and item_path.find('training_data') == -1: # training_data文件夹内的内容不要复制
            new_dest_dir = os.path.join(dest_dir, item)
            os.makedirs(new_dest_dir, exist_ok=True)  # 确保目标文件夹存在
            copy_project_files(item_path, new_dest_dir)

        # 如果是文件，并且是.py文件，则复制到目标目录
        elif os.path.isfile(item_path) and item.endswith('.py'):
            shutil.copy(item_path, dest_dir)


def _async_raise(tid, exctype):
    import inspect
    import ctypes
    """Raises an exception in the threads with id tid"""
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def tbDisplay():
    print('\033[31m Open tensorboard! \033[0m')
    subp = subprocess.Popen(
        "/home/gmm3/anaconda3/envs/torch1.13/bin/tensorboard --logdir=/home/share/hd2/Alex_Net/Blind2Unblind/training_data/ --port=6321 --bind_all",
        shell=True)  # 执行命令
    #subp.wait()



def stop_thread(thread):
    print("tensorboard threading is over")
    _async_raise(thread.ident, SystemExit)

import torch
import torch.nn.functional as F


def base_resize(tensor, base=8):
    """
    将输入的tensor进行resize，使其长和宽都小于等于原来的长和宽，并且都是8的倍数。
    """
    c, h, w = tensor.size()

    # 计算新的大小，使其满足条件：长和宽都小于等于原来的长和宽，且都是8的倍数

    new_h = (h // base) * base
    new_w = (w // base) * base

    # 使用torch的函数进行resize
    resized_tensor = F.interpolate(tensor.unsqueeze(0), size=(new_h, new_w), mode='bilinear').squeeze(0)

    return resized_tensor


def cal_color_hist(image, bins=64):
    bs, c, w, h = image.shape
    hist = torch.zeros(bs,c,bins)
    for i in range(bs):
        for j in range(c):
            hist[i][j] = torch.histc(image[i][j], bins=64) / (w * h)
    return hist

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def get_color_hist(img, bins=64, norm=True):
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    B,C,H,W = img.shape
    img = img*255
    hst = torch.zeros((B,C,bins))
    for i in range(B):
        for j in range(C):
            hst[i,j] = torch.histc(img[i,j], bins=bins, min=0, max=255)
    hist_norm = hst
    if norm:
        hist_norm = hst/(H*W)
    return hist_norm

if __name__ == '__main__':
    pth = '/home/share/hd2/image_dehaze/dataset/NH-HAZE'
