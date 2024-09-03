import math
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr



def calculate_metrics(ground_truth_folder, high_quality_folder):
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
        psnr_value = psnr(ground_truth_image, high_quality_image)
        # psnr_value = calculate_psnr(ground_truth_image, high_quality_image)
        psnr_values.append(psnr_value)
        print(f"psnr:{psnr_value}")

        # Calculate SSIM
        ssim_value = ssim(ground_truth_image, high_quality_image, multichannel=True)
        # ssim_value = calculate_ssim(ground_truth_image, high_quality_image)
        ssim_values.append(ssim_value)
        print(f"ssim:{ssim_value}")


    # Calculate the average values
    psnr_mean = np.mean(psnr_values)
    ssim_mean = np.mean(ssim_values)

    return psnr_mean, ssim_mean

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2):
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

if __name__ == '__main__':
    gt_folder_path = 'E:\\wangjing\\Fre_Aware_LLIE\\data\\Image_restoration\\LL_dataset\\LOLv1\\val\\high'
    hq_folder_path = 'E:\\wangjing\\Fre_Aware_LLIE\\results\\test\\experiment\\SGF\\SGF'

    psnr, ssim = calculate_metrics(gt_folder_path, hq_folder_path)

    print("------------------------------------------------------------")
    print("Average PSNR: {:.4f}, Average SSIM: {:.4f} ".format(psnr,ssim))


