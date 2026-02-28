import torch.optim
import torch.optim
import argparse
import time
import dataloader
from SSIM import SSIM,SSIMLOSS
import torchvision
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
import datetime
from util import *
from losses import *
import sys
from dcc_model import color_net
import numpy
# from tensorboardX import SummaryWriter

# from model_modify.new_vgg.perceptual import LossNetwork



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

    model = color_net().cuda()
    # model = nn.DataParallel(model).cuda() # 如果是用多张GPU训练，开启此语句
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # dehaze_net.load_state_dict(torch.load("/home/share/AZachary/ProjectY/MSFFDN/training_data/UIEB0426-2206/weight/Best.pth"))
    epoch_start = 0

    train_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path, mode="train")
    val_dataset = dataloader.dehazing_loader(config.orig_images_path_val, config.hazy_images_path_val, mode="val")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,
                                             num_workers=config.num_workers, pin_memory=True)



    criterion = []
    # 创建 VGG 模型和 VGGLoss 模型
    vgg = Vgg19(requires_grad=False).to('cuda:0')
    vgg_loss = VGGLoss1(device='0', vgg=vgg, normalize=False)
    criterion.append(SSIM().cuda())
    criterion.append(nn.MSELoss().cuda())
    criterion.append(vgg_loss.cuda())
    criterion.append(nn.L1Loss().cuda())
    criterion.append(nn.L1Loss().cuda())
    comput_ssim = SSIM()  # 验证指标

    model.train()
    Iters = 1

    indexX = []  # 计数损失曲线用
    indexY = []

    for epoch in range(epoch_start, epoch_start+config.num_epochs):

        if epoch <= 50:
            config.lr = 0.0003
        elif epoch <= 150:
            config.lr = 0.0002
        elif epoch <= 300:
            config.lr = 0.0001
        elif epoch <= 400:
            config.lr = 0.00003
        elif epoch <= 500:
            config.lr = 0.00001
        elif epoch <= 1000:
            config.lr = 0.000001
        # elif epoch > 60 and epoch <= 70:
        #     config.lr = 0.000003
        # elif epoch > 70 and epoch <= 200:
        #     config.lr = 0.000001
        # elif epoch > 200 and epoch <= 500:
        #     config.lr = 0.0000001
        print("now lr == %f" % config.lr)

        print("*" * 60 + 'Epoch: '+str(epoch-epoch_start) + "*" * 60)


        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8,
                                      weight_decay=0.02)


        loop = tqdm(enumerate(train_loader), total=len(train_loader), ncols=120, colour="red")  # 使用tqdm进度条重新封装
        for iteration, (gt, hazy, extension) in loop:
            #y--data_clean   x--data_hazy
            gt = gt.cuda()
            hazy = hazy.cuda()


            try:
                gray, hist, out = model(hazy)

                ssim_loss = 1-criterion[0](out,gt)
                loss = 0
                loss += ssim_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                Iters += 1

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(e)
                    torch.cuda.empty_cache()
                else:
                    raise e

            loop.set_description(f'Epoch [{epoch}/{config.num_epochs}]')
            loop.set_postfix(Loss=loss.item())

        val_ssim = []
        print("start Val!")
        # Validation Stage
        mkdir(config.sample_output_folder + f"/epochs/epoch{str(epoch).zfill(3)}")
        with torch.no_grad():
            loop_val = tqdm(enumerate(val_loader), total=len(val_loader), ncols=100,colour='green')
            for id, (val_y, val_x, extension) in loop_val:
                val_y = val_y.cuda()
                val_x = val_x.cuda()

                gray, hist, val_out = model(val_x)

                iter_ssim = comput_ssim(val_y, val_out)

                val_ssim.append(iter_ssim.item())
                torchvision.utils.save_image(torch.cat((val_x, val_y, val_out), 0),
                                             f"{config.sample_output_folder}/epochs/epoch{str(epoch).zfill(3)}/{str(id + 1).zfill(3)}{extension[0]}")

                # grid = torchvision.utils.make_grid(torch.cat((val_x, val_y, val_out), 0))
                # writer.add_image('images', grid, id)
                #
                # writer.add_scalar('val', iter_ssim.item(), global_step=epoch)
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
                now_max = np.argmax(indexY)
                indexY.append(now)
                print('max epoch %i' % now_max, 'SSIM:', indexY[now_max], 'Now Epoch mean SSIM is:', now)

                if now >= indexY[now_max]:
                    weight_name=f"best_{now:.4f}.pth"
                    torch.save(model.state_dict(), config.snapshots_folder +weight_name)
                    print("\033[31msave pth!！！！！！！！！！！！！！！！！！！！！\033[0m")
                    ssim_log = "[%i,%f]*" % (epoch, np.mean(val_ssim)) + "\n"
                if now <= indexY[now_max]:
                    shutil.rmtree(config.sample_output_folder + f"/epochs/epoch{str(epoch).zfill(3)}")

            with open('training_data/%s/train.log' % config.modelname, "a+", encoding="utf-8") as f:
                f.write(ssim_log)

if __name__ == "__main__":

    defaultname = "UIEB"+datetime.datetime.now().strftime("%m%d-%H%M")
    # defaultname = "UIEB_A01(baseline_1)"
    # todo: change defaultname
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    # Input Parameters

    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--snapshots_folder', type=str, default="training_data/%s/weight/" % defaultname)
    parser.add_argument('--sample_output_folder', type=str, default="training_data/%s/sample/" % defaultname)
    parser.add_argument('--results_output_folder', type=str, default="training_data/%s/sample/results/" % defaultname)
    parser.add_argument('--modelname', type=str, default=defaultname)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=4)

    parser.add_argument('--orig_images_path', type=str, default=r"/home/share/UIE_Datasets/UIEB-100/ref-790")
    parser.add_argument('--hazy_images_path', type=str, default=r"/home/share/UIE_Datasets/UIEB-100/raw-790")
    parser.add_argument('--orig_images_path_val', type=str,default=r"/home/share/UIE_Datasets/UIEB-100/ref-100")
    parser.add_argument('--hazy_images_path_val', type=str,default=r"/home/share/UIE_Datasets/UIEB-100/raw-100")

    parser.add_argument('--cudaid', type=str, default="0", help="choose cuda device id 0-7.")
    # todo: change cudaid and batchsize
    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cudaid
    # 对系统可见的GPU索引号，0,1,2,3...

    mkdir("training_data")

    # 把未正常训练的文件夹删除掉
    previous_training = os.listdir("training_data")
    for src_folder in previous_training:
        src_folder = os.path.join("training_data", src_folder)
        train_log_file = os.path.join(src_folder, "train.log")
        if os.path.exists(train_log_file):
            with open(train_log_file, 'r') as file:
                line_count =  sum(1 for line in file)
            if line_count < 2:
                try:
                    shutil.rmtree(src_folder)
                    print(f"文件夹 '{src_folder}' 已删除。")
                except OSError as e:
                    print(f"删除文件夹 '{src_folder}' 失败: {e}")
        else:
            # print(f"文件 '{train_log_file}' 不存在。")
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

    with open('training_data/%s/project_files/%s.txt' % (config.modelname, config.modelname), "w") as f:  # 设置文件对象
        for i in vars(config):
            f.write(i + ":" + str(vars(config)[i]) + '\n')
        f.write("train time:%s"% str(current_time))

    # transfPY(config)
    copy_project_files(os.getcwd(), 'training_data/%s/project_files' % config.modelname)
    # thread = threading.Thread(target=tbDisplay)
    # thread.start()
    time.sleep(2)


    try:
        train(config)
        sys.exit()

    except Exception as e:
        raise(e)
        torch.cuda.empty_cache()
        time.sleep(2)
        sys.exit()

