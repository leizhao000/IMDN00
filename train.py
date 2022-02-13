import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.MDASR import mdasrcnew
from model import architecture
from data import DIV2K,Set5_val
import utils
import random
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import skimage.color as sc


# 训练参数设置
parser = argparse.ArgumentParser(description="MDASR")
parser.add_argument("--batch_size", type=int, default=16,
                    help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1,
                    help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=1000,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=5e-4,
                    help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=200,
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5,
                    help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="use cuda")
parser.add_argument("--resume", default="", type=str,
                    help="path to checkpoint")
parser.add_argument("--start_epoch", default=1, type=int,
                    help="manual epoch number")
parser.add_argument("--threads", type=int, default=8,
                    help="number of threads for data loading")
parser.add_argument("--root", type=str, default="./dataset",
                    help='dataset directory')
parser.add_argument("--n_train", type=int, default=800,
                    help="number of training set")
parser.add_argument("--n_val", type=int, default=1,
                    help="number of validation set")
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--scale", type=int, default=2,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=192,
                    help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--pretrained", default="", type=str,
                    help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=True)
parser.add_argument("--ext", type=str, default='.npy')
parser.add_argument("--phase", type=str, default='train')


args = parser.parse_args()
print(args)
torch.backends.cudnn.benchmark = True

#设置随机种子,确保训练时抽取的数据集数据一致
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print("Ramdom Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

#使用cuda
cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')
ngpu=2

#调用tensorboard进行实时监控，打开监控命令：tensorboard --logdir runs
writer = SummaryWriter()


print("===> Loading datasets")
trainset=DIV2K.div2k(args)#scale,root,ext
testset = Set5_val.DatasetFromFolderVal("data_image/Set5/",
                                       "data_image/Set5_LR/x2/",
                                       args.scale)
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)

print("===> Building models")
args.is_train = True
model = mdasrcnew.MDASR()
l1_criterion = nn.L1Loss() #用l1 loss

print("===> Setting GPU")
if cuda:#迁移至默认设备进行训练
    model = model.to(device)
    l1_criterion = l1_criterion.to(device)

print("===> Setting Optimizer")#初始化优化器
optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.pretrained:
    if os.path.isfile(args.pretrained):
        print("===> loading models '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        args.start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("===> no models found at '{}'".format(args.pretrained))

model = nn.DataParallel(model, device_ids=list(range(ngpu)))


#定义train函数
def train(epoch):
    model.train()
    global writer
    utils.adjust_learning_rate(optimizer=optimizer, epoch=epoch, step_size=args.step_size, lr_init=args.lr, gamma=args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])

    loss_epoch = utils.AverageMeter()  # 统计损失函数
    n_iter = len(training_data_loader)

    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):
        if args.cuda:
            lr_tensor=lr_tensor.to(device)#将lr，hr迁移至默认设备进行训练
            hr_tensor=hr_tensor.to(device)
        #梯度下降法
        optimizer.zero_grad()  #梯度初始化，将loss关于weight的导数变成0.
        sr_tensor = model(lr_tensor) #前向传播，求出预测的值
        loss_l1 = l1_criterion(sr_tensor, hr_tensor)#求出loss
        loss_sr=loss_l1
        loss_sr.backward()#反向传播求梯度
        optimizer.step()#更新所有参数
        loss_epoch.update(loss_sr.item(),lr_tensor.size(0))
        #监控图像变化
        if iteration==(n_iter-2):
            writer.add_image('MDASR/epoch_' + str(epoch) + '_1',
                             make_grid(lr_tensor[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)
            writer.add_image('MDASR/epoch_' + str(epoch) + '_2',
                             make_grid(sr_tensor[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)
            writer.add_image('MDASR/epoch_' + str(epoch) + '_3',
                             make_grid(hr_tensor[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)

        # 打印结果
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss_l1: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                  loss_sr.item()))
        # 手动释放内存
    del lr_tensor, hr_tensor, sr_tensor
        # 监控损失值变化
    writer.add_scalar('MDASR/Loss', loss_epoch.val, epoch)

def valid():
    model.eval()

    avg_psnr, avg_ssim = 0, 0
    for batch in testing_data_loader:
        lr_tensor, hr_tensor = batch[0], batch[1]
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

        with torch.no_grad():
            pre = model(lr_tensor)

        sr_img = utils.tensor2np(pre.detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        if args.isY is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        avg_psnr += utils.compute_psnr(im_pre, im_label)
        avg_ssim += utils.compute_ssim(im_pre, im_label)
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader), avg_ssim / len(testing_data_loader)))



def save_checkpoint(epoch):
    model_folder = "checkpoints/result/MDASR0105_x{}/".format(args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder) \
            :
        os.makedirs(model_folder)
    torch.save({
            'epoch': epoch,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict()
        }, model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


print("===> Training")
print_network(model)
for epoch in range(args.start_epoch, args.nEpochs + 1):
    valid()
    train(epoch)
    save_checkpoint(epoch)




