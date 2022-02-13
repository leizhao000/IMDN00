import argparse
import torch
import torch.nn as nn
import os
import numpy as np
import utils
import skimage.color as sc
import cv2
from model.MDASR import mdasrcnew
from model import architecture
# Testing settings


parser = argparse.ArgumentParser(description='MDASR')
parser.add_argument("--test_hr_folder", type=str, default='data_image/Set5/')
parser.add_argument("--test_lr_folder", type=str, default='data_image/Set5_LR/x2/',
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=str, default='results/Set5/x2')
parser.add_argument("--checkpoint", type=str, default='checkpoints/result/MDASRnew1_C52_x2/epoch_1000.pth',
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--upscale_factor", type=int, default=2,
                    help='upscaling factor')
parser.add_argument("--is_y", action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if cuda else 'cpu')

filepath = opt.test_hr_folder
if filepath.split('/')[-2] == 'Set5' or filepath.split('/')[-2] == 'Set14':
    ext = '.bmp'
else:
    ext = '.png'

filelist = utils.get_list(filepath, ext=ext)
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))

checkpoint=torch.load(opt.checkpoint)
model = mdasrcnew.MDASR()
model = model.to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

#model=nn.DataParallel(model,device_ids=list(range(2)))
#model = model.to(device)
#model_dict = utils.load_state_dict(opt.checkpoint)
#model.load_state_dict(model_dict, strict=True)


i = 0
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for imname in filelist:
    im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    im_gt = utils.modcrop(im_gt, opt.upscale_factor)
    im_l = cv2.imread(opt.test_lr_folder + imname.split('/')[-1].split('.')[0] + 'x' + str(opt.upscale_factor) + ext, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    if len(im_gt.shape) < 3:
        im_gt = im_gt[..., np.newaxis]
        im_gt = np.concatenate([im_gt] * 3, 2)
        im_l = im_l[..., np.newaxis]
        im_l = np.concatenate([im_l] * 3, 2)
    im_input = im_l / 255.0
    im_input = np.transpose(im_input, (2, 0, 1))
    im_input = im_input[np.newaxis, ...]
    im_input = torch.from_numpy(im_input).float()

    if cuda:
        model = model.to(device)
        im_input = im_input.to(device)

    with torch.no_grad():
        start.record()
        out = model(im_input)
        end.record()
        torch.cuda.synchronize()
        time_list[i] = start.elapsed_time(end)  # milliseconds

    out_img = utils.tensor2np(out.detach()[0])
    crop_size = opt.upscale_factor
    cropped_sr_img = utils.shave(out_img, crop_size)
    cropped_gt_img = utils.shave(im_gt, crop_size)
    if opt.is_y is True:
        im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
        im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
    else:
        im_label = cropped_gt_img
        im_pre = cropped_sr_img
    psnr_list[i] = utils.compute_psnr(im_pre, im_label)
    ssim_list[i] = utils.compute_ssim(im_pre, im_label)


    output_folder = os.path.join(opt.output_folder,
                                 imname.split('/')[-1].split('.')[0] + 'x' + str(opt.upscale_factor) + '.png')

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    cv2.imwrite(output_folder, out_img[:, :, [2, 1, 0]])
    i += 1


print("Mean PSNR: {}, SSIM: {}, TIME: {} ms".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))