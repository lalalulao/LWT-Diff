import datetime
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import make_grid

import utils
from models.unet import DiffusionUNet
from models.wavelet import DWT, IWT


# from LL_Unet_src.unet import UNet as SegmentUnet
from Fusion_model import BiGFF
from models.mods import DRM, RCM

from pytorch_msssim import ssim
import cv2
from utils.logging import save_image

from torchvision.models.vgg import vgg16
import torchvision.models as models


# 对输入的数据进行转换，将数据归一化到【-1,1】
def data_transform(X):
    return 2 * X - 1.0

# 数据的逆转换:将数据逆向转换，[0.,1]
def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

# Total Variation ：总变分损失
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

vgg = models.vgg16(pretrained=True).features.cuda()
for param in vgg.parameters():
    param.requires_grad_(False)

# 定义VGG loss函数
class vgg_loss(nn.Module):
    def __init__(self):
        super(vgg_loss, self).__init__()

    def forward(self, x, y):
        features_x = vgg(x)
        features_y = vgg(y)
        loss = torch.mean((features_x - features_y) ** 2)
        return loss


# 模型的指数移动平均：模型参数平滑技术，用于提高模型的泛化能力和稳定性
class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

# 网络结构
class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device
        self.detail_retain = DRM(64)
        self.Unet = DiffusionUNet(config)


        self.conv3_64 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        self.bigff = BiGFF(in_channels=64, out_channels=64)

        self.fusion_layer1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.enhance_fusion = RCM(in_channels=3, out_channels=3)

        # β
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):  # 计算α，用于生成噪声
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    # 用于训练时的采样方法：去噪
    def sample_training(self, x_cond, b, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape

        seq_next = [-1] + list(seq[:-1])

        x = torch.randn(n, c, h, w, device=self.device)

        xs = [x]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]

    def forward(self, x):
        data_dict = {}
        dwt, idwt = DWT(), IWT()

        input_img = x[:, :3, :, :]
        n, c, h, w = input_img.shape

        detail_64 = self.conv3_64(input_img)
        detail_64 = self.detail_retain(detail_64)

        input_img_norm = data_transform(input_img)
        input_dwt = dwt(input_img_norm)

        b = self.betas.to(input_img.device)

        t = torch.randint(low=0, high=self.num_timesteps, size=(input_dwt.shape[0] // 2 + 1,)).to(
            self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_dwt.shape[0]].to(
            x.device)

        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        e = torch.randn_like(input_dwt)

        if self.training:
            gt_img_norm = data_transform(x[:, 3:, :, :])
            gt_dwt = dwt(gt_img_norm)
            x = gt_dwt * a.sqrt() + e * (1.0 - a).sqrt()
            noise_output = self.Unet(torch.cat([input_dwt, x], dim=1), t.float())
            denoise_dwt = self.sample_training(input_dwt, b)

            pred_ddpm = idwt(denoise_dwt)
            pred_ddpm = inverse_data_transform(pred_ddpm)

            # 增加图像融合的结果
            pred_ddpm_64 = self.conv3_64(pred_ddpm)
            bigff_img = self.bigff(detail_64, pred_ddpm_64)

            pred_x = self.fusion_layer1(bigff_img)    # 64
            pred_x = self.enhance_fusion(pred_x)

            data_dict["gt_dwt"] = gt_dwt
            data_dict["denoise_dwt"] = denoise_dwt
            data_dict["noise_output"] = noise_output
            data_dict["e"] = e
            data_dict["detail"] = detail_64
            data_dict["pred_ddpm"] = pred_ddpm
            data_dict["pred_x"] = pred_x


        else:
            denoise_dwt = self.sample_training(input_dwt, b)
            pred_ddpm = idwt(denoise_dwt)
            pred_ddpm = inverse_data_transform(pred_ddpm)

            # 增加图像融合的结果
            pred_ddpm_64 = self.conv3_64(pred_ddpm)
            bigff_img = self.bigff(detail_64, pred_ddpm_64)

            pred_x = self.fusion_layer1(bigff_img)  # 64
            pred_x = self.enhance_fusion(pred_x)

            data_dict["pred_ddpm"] = pred_ddpm
            data_dict["detail"] = detail_64
            data_dict["pred_x"] = pred_x

        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        # 损失函数：均方误差，L1loss,TVloss
        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.TV_loss = TVLoss()
        #self.VGG_loss = vgg_loss()

        self.optimizer, self.scheduler = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        max_psnr = 0
        max_ssim = 0
        max_psnr_step = 0
        max_ssim_step = 0

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)

                output = self.model(x)

                noise_loss, photo_loss, frequency_loss = self.estimation_loss(x, output)

                #loss = noise_loss + photo_loss + frequency_loss
                loss = noise_loss + photo_loss + frequency_loss

                if self.step % 10 == 0:
                    print("time:{}, step:{}, lr:{:.6f}, noise_loss:{:.4f}, photo_loss:{:.4f}, "
                          "frequency_loss:{:.4f}".format(timestamp, self.step, self.scheduler.get_last_lr()[0],
                                                         noise_loss.item(), photo_loss.item(),
                                                         frequency_loss.item()))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)
                    psnr, ssim = self.calculate_psnr_ssim()

                    print("time:{}, step:{}, Average PSNR: {:.4f}, Average SSIM: {:.4f} ".format(timestamp, self.step, psnr, ssim))

                    #打印最大psnr和ssim
                    if psnr > max_psnr:
                        max_psnr = psnr
                        max_psnr_step = self.step
                    if ssim > max_ssim:
                        max_ssim = ssim
                        max_ssim_step = self.step
                    print("Max PSNR: {:.4f}, Max SSIM: {:.4f}, Max psnr Step: {:.4f}, Max ssim Step: {:.4f} ".format(max_psnr, max_ssim, max_psnr_step, max_ssim_step))

                    utils.logging.save_checkpoint({'step': self.step,
                                                   'epoch': epoch + 1,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'scheduler': self.scheduler.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                  filename=os.path.join(self.config.data.ckpt_dir, f'{self.step}_model_latest'))

            self.scheduler.step()


    def estimation_loss(self, x, output):
        noise_output, e, denoise_dwt, gt_dwt, pred_ddpm, pred_x = output["noise_output"],output["e"],output["denoise_dwt"],output["gt_dwt"],output["pred_ddpm"], output["pred_x"]


        gt_img = x[:, 3:, :, :].to(self.device)
        noise_loss = self.l2_loss(noise_output, e)
        frequency_loss = 1 * (self.l2_loss(denoise_dwt, gt_dwt)) + 0.8 * (self.TV_loss(denoise_dwt))
        content_loss = self.l1_loss(pred_x, gt_img)
        ssim_loss = 1 - ssim(pred_x, gt_img, data_range=1.0).to(self.device)
        photo_loss = content_loss + ssim_loss

        return noise_loss, photo_loss, frequency_loss

    def sample_validation_patches(self, val_loader, step):
        self.model.eval()
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):

                b, _, img_h, img_w = x.shape
                img_h_32 = int(32 * np.ceil(img_h / 32.0))
                img_w_32 = int(32 * np.ceil(img_w / 32.0))
                x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')

                out = self.model(x.to(self.device))
                pred_x = out["pred_x"]
                pred_x = pred_x[:, :, :img_h, :img_w]

                image_number = int(y[0].rsplit('\\', 1)[-1])
                # 要保存的目录路径
                save_path = self.args.image_folder
                # 构建保存图像的完整路径
                image_name = f"{image_number}.png"
                image_path = os.path.join(save_path, image_name)
                # 保存图像
                save_image(pred_x, image_path)

    def calculate_psnr_ssim(self):
        gt_folder_path = 'E:\\data\\Image_restoration\\LL_dataset\\LOLv1\\val\\high'
        hq_folder_path = 'E:\\results\\train\\LOLV1'
        psnr, ssim = self.calculate_metrics(gt_folder_path, hq_folder_path)

        return psnr, ssim

    def calculate_metrics(self, ground_truth_folder, high_quality_folder):
        psnr_values = []
        ssim_values = []

        for image_name in os.listdir(high_quality_folder):
            image_path = os.path.join(high_quality_folder, image_name)
            high_quality_image = cv2.imread(image_path)

            ground_truth_image_path = os.path.join(ground_truth_folder, image_name)
            ground_truth_image = cv2.imread(ground_truth_image_path)

            # Check if the images are valid
            if high_quality_image is None or ground_truth_image is None:
                print("Invalid image: {}".format(image_name))
                continue

            # Check if the images have the same size
            if high_quality_image.shape != ground_truth_image.shape:
                print("Images have different shapes: {}".format(image_name))
                continue

            # Calculate PSNR
            psnr_value = self.calculate_psnr(ground_truth_image, high_quality_image)
            psnr_values.append(psnr_value)

            # Calculate SSIM
            ssim_value = self.calculate_ssim(ground_truth_image, high_quality_image)
            ssim_values.append(ssim_value)

        # Calculate the average values
        psnr_mean = np.mean(psnr_values)
        ssim_mean = np.mean(ssim_values)

        return psnr_mean, ssim_mean

    def calculate_psnr(self, img1, img2):
        # img1 and img2 have range [0, 255]
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calculate_ssim(self, img1, img2):
        # img1 and img2 have range [0, 255]
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()