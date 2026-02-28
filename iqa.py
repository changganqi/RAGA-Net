from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import cv2
import glob
from PIL import Image
import numpy as np

enh="/home/s_yzm/cqq/ba/results/EUVP_GAWT"
#enh="/data1/cqq/my_project/results/LOL"
# GT="/home/s_yzm/cqq/dataset/UIEB-100/test/ref"
GT="/home/s_yzm/cqq/dataset/EUVP-200/test/ref"


def cut(img):
    if img.shape[0] != 240 or img.shape[1] != 320:
        i = img.shape[0]
        j = img.shape[1]
        if img.shape[0] != 240:
            i = 240
        if img.shape[1] != 320:
            j = 320
        img = cv2.resize(img, (j, i))
    return img
def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)

list1=glob.glob(enh + "/*")
list2=glob.glob(GT + "/*")
a=0
psnr1=0
ssim1=0
mae1=0
top_10_images = []

for enh_img,GT_img in zip(list1,list2):

    img1=cv2.imread(enh_img)
    img2=cv2.imread(GT_img)
    # img1=cut(img1)
    # img2=cut(img2)

    ssim = compare_ssim(img1, img2, channel_axis=2,win_size=11, data_range=255, multichannel=True)
    psnr = compare_psnr(img1, img2, data_range=255)
    mae =  compare_mae(img1, img2)
    top_10_images.append((enh_img.split("/")[-1], ssim))
    psnr1+=psnr
    ssim1+=ssim
    mae1+=mae
    print(enh_img.split("/")[-1] + '的psnr' + str(psnr) + '******的ssim:' + str(ssim)+'*******的MAE:'+ str(mae))
    a=a+1
    if ssim>0.9:
        print("\033[31m" + enh_img.split("/")[-1]+'的ssim值大于0.9'+ "\033[0m")

ssim=ssim1/a
psnr=psnr1/a
mae = mae1/a
print('一共',a,'张图片')
# print('平均SSIM和PSNR为 {:.4f}/{:.4f}'.format(ssim, psnr))
# print('mae = ',mae)
print('平均SSIM和PSNR, mae 为 {:.4f}/{:.4f}/{:.4f}'.format(ssim, psnr, mae * 100))
print("SSIM:{:.4f}".format(ssim))
print("PSNR:{:.4f}".format(psnr))
print("MAE:{:.4f}".format(mae* 100))

print(enh)
top_10_images = sorted(top_10_images, key=lambda x: x[1], reverse=True)

# for i in range(10):
#     file_name = top_10_images[i][0]
#     ssim_value = top_10_images[i][1]
#     print(f"第 {i+1} 名图片：{file_name}，SSIM 值：{ssim_value}")
