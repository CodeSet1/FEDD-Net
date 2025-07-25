import torch
from PIL import Image
from torchvision import transforms
from model.DTCWT.FDN_network import DecomNet as create_model
import numpy as np
import cv2
import os


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.ToTensor()])        #数据转换，用于将多个转换操作组合在一起，transforms.ToTensor()将图像转换为张量

    root = ""
    assert os.path.exists(root), "file: '{}' dose not exist.".format(root)      #检查文件夹是否存在

    images_path=loadfiles(root=root)        #加载文件夹下的所有图片路径
    for index in range(len(images_path)):
        assert os.path.exists(images_path[index]), "file: '{}' dose not exist.".format(images_path[index])      #验证文件路径
    print("path checking complete!")
    print("confirmly find {} images for computing".format(len(images_path)))        #打印信息，确认找到的图片数量

    model = create_model().to(device)
    model_weight_path = "./weights/checkpoint_LOL_FDN.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device)['model'])
    model.eval()        #加载模型
    for img_path in images_path:
        img = Image.open(img_path)
        img = resize(img)
        img = data_transform(img)
        img = img.unsqueeze(0)      #处理图像，在批次维度上增加一个维度，使其形状变为（1,C,H,W）
        with torch.no_grad():
            R, L = (model(img.to(device)))      #使用torch.no_grad()上下文管理器关闭梯度计算，以节省内容核计算资源，将图像输入模型，得到两个输出R和L


        R = R.squeeze(0).detach().cpu().numpy()     #squeeze（）去掉批次维度
        L = torch.cat([L,L,L],dim=1)        #在通道维度上重复三次
        L = L.squeeze(0).detach().cpu().numpy()     #detach（）从计算图中分离张量，防止梯度计算
        R = np.transpose(R,(1,2,0))
        L = np.transpose(L,(1,2,0))     #调整数组维度，使其符合图像格式
        name=getnameindex(img_path)     #获取图像的名称或索引
        savepic(R, name, flag="R")
        savepic(L, name, flag="L")

def savepic(outputpic, name, flag):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()       #UMat加速图像处理
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)     #归一化图像，将图像数据归一化0到255之间，以便保存8位图像
    outputpic=outputpic[:, :, ::-1]     #BGR转RGB

    root = "./results/LOL_high_eval"
    root_path = os.path.join(root, flag)        #创建保存路径

    if os.path.exists("./results") is False:
        os.makedirs("./results")
    if os.path.exists(root) is False:
        os.makedirs(root)
    if os.path.exists(root_path) is False:
        os.makedirs(root_path)      #检查并创建必要的目录，如果目录不存在则创建
    path = root_path + "/{}.png".format(name)
    cv2.imwrite(path, outputpic)        #保存图像
    assert os.path.exists(path), "file: '{}' dose not exist.".format(path)      #断言文件存在，如果文件不存在则抛出异常
    print("complete compute {}.png and save".format(name))      #打印保存完成的提示信息

def loadfiles(root):
    images_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP"]
    images = [os.path.join(root, i) for i in os.listdir(root)
              if os.path.splitext(i)[-1] in supported]
    for index in range(len(images)):
        img_path = images[index]
        images_path.append(img_path)

    print("find {} images for computing.".format(len(images_path)))
    return images_path

def getnameindex(path):
    assert os.path.exists(path), "file: '{}' dose not exist.".format(path)
    path = path.replace("\\", "/")
    label = path.split("/")[-1].split(".")[0]
    return label

def resize(image):      #调整图像大小，使其宽度和高度都是8的倍数
    original_width, original_height = image.size        #获取图像的原始宽度和高度

    new_width = original_width - (original_width % 8)       #计算新的宽度和高度，使其为8的倍数
    new_height = original_height - (original_height % 8)
    resized_image = image.resize((new_width, new_height))       #调整图像大小
    return resized_image

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()