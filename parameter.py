import numpy as np
import math
import cv2
from scipy.ndimage import distance_transform_edt
import statistics

def meansquareerror(src, dst):
    if src.ndim == 3:
        src = src.mean(2)
        #dst = dst.ndim(2)
    mse = np.mean((src - dst) ** 2)
    return mse

def signaltonoise(src):
    a = np.asanyarray(src)
    m = np.mean(a)
    sd = np.std(a)
    return abs(10 * math.log10(math.pow(m,2) / math.pow(sd,2)))

def psnr(src, dst):
    if src.ndim == 3:
        src = src.mean(2)
        #dst = dst.mean(2)
    mse = np.mean((src - dst) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX =255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

DEFAULT_ALPHA = 1.0 / 9

def fom(img, img_gold_std, alpha=DEFAULT_ALPHA):
    """
    Computes Pratt's Figure of Merit for the given image img, using a gold
    standard image as source of the ideal edge pixels.
    """

    # To avoid oversmoothing, we apply canny edge detection with very low
    # standard deviation of the Gaussian kernel (sigma = 0.1).
    edges_img = cv2.Canny(img, 20, 50)
    edges_gold = cv2.Canny(img_gold_std, 20, 50)

    # Compute the distance transform for the gold standard image.
    dist = distance_transform_edt(np.invert(edges_gold))

    fom = 1.0 / np.maximum(
        np.count_nonzero(edges_img),
        np.count_nonzero(edges_gold))

    N, M = img.shape

    for i in range(0, N):
        for j in range(0, M):
            if edges_img[i, j]:
                fom += 1.0 / (1.0 + dist[i, j] * dist[i, j] * alpha)

    fom /= np.maximum(
        np.count_nonzero(edges_img),
        np.count_nonzero(edges_gold))

    return fom