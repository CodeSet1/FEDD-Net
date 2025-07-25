import os
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from my_dataset import MyDataSet

from model.DTCWT.FDN_network import DecomNet as create_model
from utils import read_data, train_one_epoch, evaluate, create_lr_scheduler
import datetime

import transforms as T

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./experiments") is False:
        os.makedirs("./experiments")

    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = "./experiments/FDN_train_{}".format(file_name)
    os.makedirs(filefold_path)
    file_img_path = os.path.join(filefold_path, "img")
    os.makedirs(file_img_path)
    file_weights_path = os.path.join(filefold_path, "weights")
    os.makedirs(file_weights_path)
    file_log_path = os.path.join(filefold_path, "log")
    os.makedirs(file_log_path)

    tb_writer = SummaryWriter(log_dir=file_log_path)

    best_valloss = 1e5
    start_epoch = 0

    train_low_path, train_high_path, val_low_path, val_high_path = read_data(args.data_path)

    data_transform = {
        "train": T.Compose([T.RandomCrop(128),
                            T.RandomHorizontalFlip(0.5),
                            T.RandomVerticalFlip(0.5),
                            T.ToTensor()]),

        "val": T.Compose([T.ToTensor()])}

    train_dataset = MyDataSet(images_low_path=train_low_path,
                              images_high_path=train_high_path,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_low_path=val_low_path,
                            images_high_path=val_high_path,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model().to(device)   #创建模型实例，to(device)将模型移动到指定的设备上
    if args.use_dp == True:             #检查是否启用分布式数据并行，
        model = torch.nn.DataParallel(model).cuda()  #如果启用了分布式数据并行，则使用 torch.nn.DataParallel 将模型封装起来

###############################从指定的路径加载预训练的权重文件###############################
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)  #检查权重文件是否存在
        weights_dict = torch.load(args.weights, map_location=device)["model"]                      #加载权重文件，存储在weights_dict中
        print(model.load_state_dict(weights_dict, strict=False))                                   #strict=False参数表示在加载权重时，允许模型和权重文件中的参数数量不一致。如果设置为True，则要求模型和权重文件中的参数数量必须完全一致，否则会抛出错误。
###########################################################################################

    pg = [p for p in model.parameters() if p.requires_grad]                                     #通过列表推导式从模型的所有参数中选择那些需要计算梯度的参数（即 requires_grad=True 的参数）。这些参数将被用于优化器的更新。
    optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5E-5)    #初始化优化器
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)  #创建学习率调度器，用于在训练过程中动态调整学习率

    if args.resume:    #从检查点恢复训练
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.epochs):
        # train
        train_loss, train_rec_loss, train_equal_R_loss, \
        train_smooth_high_loss, lr = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                lr_scheduler=lr_scheduler,
                                                device=device,
                                                epoch=epoch, filefold_path=file_img_path)               #训练模型，调用train_one_epoch函数进行一次完整的训练过程，并返回训练过程中的损失值和当前学习率

        # validate
        val_loss, val_rec_loss, val_equal_R_loss, \
        val_smooth_high_loss = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch, lr=lr, filefold_path=file_img_path)  #验证模型，调用evaluate函数进行一次完整的验证过程，并返回验证过程中的损失值

        tb_writer.add_scalar("train_total_loss", train_loss, epoch)
        tb_writer.add_scalar("train_rec_loss", train_rec_loss, epoch)
        tb_writer.add_scalar("train_equal_R_loss", train_equal_R_loss, epoch)
        tb_writer.add_scalar("train_smooth_high_loss", train_smooth_high_loss, epoch)

        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("val_rec_loss", val_rec_loss, epoch)
        tb_writer.add_scalar("val_equal_R_loss", val_equal_R_loss, epoch)
        tb_writer.add_scalar("val_smooth_high_loss", val_smooth_high_loss, epoch)      #使用TensorBoard的tb_writer记录训练和验证过程中的损失值,以便后续可视化


        if val_loss < best_valloss:
            if args.use_dp == True:
                save_file = {"model": model.module.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
            else:
                save_file = {"model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
            torch.save(save_file, file_weights_path + "/" + "checkpoint_FDN.pth")
            best_valloss = val_loss                                                    #保存最佳模型，如果当前验证损失val_loss小于之前保存的最佳模型的验证损失best_valloss，则保存当前的模型状态并更新

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data-path', type=str,
                        default="dataset/LOL")
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--use_dp', default=False, help='use dp-multigpus')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')
    opt = parser.parse_args()

    main(opt)