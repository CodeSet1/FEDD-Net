from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):                    #定义MyDataSet的类，该类继承自Dataset类，用于数据集加载
    def __init__(self, images_low_path: list, images_high_path: list, transform=None):
        self.images_low_path = images_low_path
        self.images_high_path = images_high_path
        self.transform = transform           #类定义和初始化方法，它接受三个参数：images_low_path\images_high_path\transform(一个可选的图像变换函数)

    def __len__(self):
        return len(self.images_low_path)     #_len_方法返回数据集的长度，即low image的数量

    def __getitem__(self, item):             #获取数据集的某一项
        img = Image.open(self.images_low_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_low_path[item]))
        img_ref = Image.open(self.images_high_path[item])
        if img_ref.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_high_path[item]))

        if self.transform is not None:
            img, img_ref = self.transform(img, img_ref)

        return img, img_ref                 #transform对图像进行相应的变换，最后返回low和high的图像对

    @staticmethod     #静态方法
    def collate_fn(batch):                  #用于处理一批数据，cllate_fn将一批数据中的图像和参考图像分别堆叠起来，以便后续处理
        images, images_ref = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        images_ref = torch.stack(images_ref, dim=0)   #数据通常以小批量batch的形式输入模型，而每个小批量中的数据需要被处理成相同的张量
        return images, images_ref
