import torch
import logging
from tensorboardX import SummaryWriter
import argparse
import core.logger as Logger
import numpy as np
import cv2
import os
import dataloader as Data
import model.RDR.core.metrics as Metrics

def normal_to_miuns1_1(x):
    return x * 2.0 - 1.0

def output_img(outputpic):
    outputpic[outputpic > 1.] = 1.
    outputpic[outputpic < 0.] = 0.
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255.0, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic = outputpic[:, :, ::-1]
    outputpic = cv2.cvtColor(outputpic, cv2.COLOR_BGR2RGB)
    return outputpic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/gdrdnet_val.json',
                        help='JSON file for configuration')
    parser.add_argument('--config_RDR', type=str, default='model/RDR/config/RDR_data_val.json',
                        help='JSON file for configuration')
    parser.add_argument('--config_IDA', type=str, default='model/IDA/config/IDA_data_val.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    # the gt-means is not using by default now, check the need.
    parser.add_argument('--use_gtmeans', type=bool, default=False)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_infer', action='store_true')

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parse configs
    args = parser.parse_args()
    opt, opt_RDA, opt_IDA = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    opt_RDA = Logger.dict_to_nonedict(opt_RDA)
    opt_IDA = Logger.dict_to_nonedict(opt_IDA)

    # Change to generate results with seed
    torch.manual_seed(1)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)

    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    import model.RDR.model as Model_RDR
    diffusion_RDR = Model_RDR.create_model_val(opt_RDA)
    logger.info('Initial RDA Model Finished')

    import model.IDA.model as Model_IDA
    diffusion_IDA = Model_IDA.create_model(opt_IDA)
    logger.info('Initial IDA Model Finished')

    diffusion_RDR.set_new_noise_schedule(
        opt_RDA['model']['beta_schedule']['val'], schedule_phase='val')

    diffusion_IDA.set_new_noise_schedule(
        opt_IDA['model']['beta_schedule']['val'], schedule_phase='val')

    logger.info('Begin Model Inference.')
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    from model.DTCWT.FDN_network import DecomNet as create_model
    model_TDN = create_model().to(device)
    model_TDN_weight_path = "model/DTCWT/weights/checkpoint_FDN.pth"
    model_TDN.load_state_dict(torch.load(model_TDN_weight_path, map_location=device)['model'])
    model_TDN.eval()

    for _, val_data_collect in enumerate(val_loader):
        name = str(val_data_collect[1][0])
        val_data = val_data_collect[0]

        with torch.no_grad():
            R, L = (model_TDN(val_data['low'].to(device)))
        R = R.squeeze(0).detach().cpu().numpy()
        L = torch.cat([L, L, L],dim=1)
        L = L.squeeze(0).detach().cpu().numpy()
        R = np.transpose(R, (1, 2, 0))
        L = np.transpose(L, (1, 2, 0))

        R_img = output_img(R)
        L_img = output_img(L)

        import torchvision
        totensor = torchvision.transforms.ToTensor()
        R = totensor(R_img.copy())/255.
        L = totensor(L_img.copy())/255.

        R = normal_to_miuns1_1(R).unsqueeze(0)
        L = normal_to_miuns1_1(L).mean(dim=0).unsqueeze(0).unsqueeze(0)

        val_RDA_data = {'low': R}
        diffusion_RDR.feed_data(val_RDA_data)
        diffusion_RDR.test_ddim(continous=True)

        val_IDA_data = {'low': L}
        diffusion_IDA.feed_data(val_IDA_data)
        diffusion_IDA.test_ddim(continous=True)

        visuals_RDA = diffusion_RDR.get_current_visuals_val()
        visuals_IDA = diffusion_IDA.get_current_visuals_val()

        G_R = Metrics.tensor2img(visuals_RDA['DDPM'][-1])
        G_L = Metrics.tensor2img(visuals_IDA['DDPM'].squeeze(0)[-1])
        I = ((G_R / 255.0 * G_L[:, :, np.newaxis] / 255.0) * 255.0).astype(np.uint8)

        # some method get very high PSNR (more than 27), check this in the code
        if args.use_gtmeans is True:
            normal_I = val_data['high'].detach().cpu().numpy().squeeze(0)*255.0
            normal_I_means = np.mean(normal_I)
            pred_I_means = np.mean(I)
            I = I * normal_I_means / pred_I_means

        Metrics.save_img(I, '{}/{}.png'.format(result_path, name))

    logger.info('Results are saved in {}'.format(opt['path']['results']))