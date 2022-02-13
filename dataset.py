import os
from PIL import Image


def resize(old_path, gt_path,new_path, resample):
    """
     通过指定的resample方式调整old_path下所有的jpg图片为指定的size，并保存到new_path下
    """
    if os.path.isdir(old_path):
        for child in os.listdir(old_path):

            if child.find('.png') > 0:
                im = Image.open(old_path + child)
                a=(im.width//12)*12
                b=(im.height//12)*12
                im = im.crop((0,0,a,b))
                im.save(gt_path+child,im.format)
                im_resized = im.resize((int(im.width /2), int(im.height /2)), resample=resample)
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                if not os.path.exists(gt_path):
                    os.makedirs(gt_path)
                print(child, 'resize successfully!')
                im_resized.save(new_path + child, im.format)
            child_path = old_path + child + '/'

            resize(child_path, gt_path,new_path, resample)


if __name__ == "__main__":
    old_path = './data_image/mura/'
    gt_path='./data_image/mura_train/'
    new_path = './data_image/mura_test/'
    resample = Image.BICUBIC # 使用线性插值法重采样
    resize(old_path, gt_path,new_path, resample)

