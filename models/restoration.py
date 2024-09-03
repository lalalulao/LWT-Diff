import torch
import numpy as np
import utils
import os
import torch.nn.functional as F

from utils.logging import save_image
import re

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            # self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.load_ddm_ckpt(args.resume)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader):
        self.diffusion.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                b, c, h, w = x_cond.shape
                img_h_32 = int(32 * np.ceil(h / 32.0))
                img_w_32 = int(32 * np.ceil(w / 32.0))
                x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                x_output = self.diffusive_restoration(x_cond)
                x_output = x_output[:, :, :h, :w]
                # 从路径中提取数字部分
                image_number = int(y[0].rsplit('\\', 1)[-1])

                # 要保存的目录路径
                save_path = self.args.image_folder

                # 构建保存图像的完整路径
                image_name = f"{image_number}.png"
                image_path = os.path.join(save_path, image_name)

                # 保存图像
                save_image(x_output, image_path)

                # 原本代码：
                # print(f"processing image {y[0]}")
                print(f"processing image {image_name}")             #打印输出

    def diffusive_restoration(self, x_cond):
        # x_output = self.diffusion.model(x_cond.to(self.device))
        x_output = self.diffusion.model(x_cond)

        return x_output["pred_x"]

