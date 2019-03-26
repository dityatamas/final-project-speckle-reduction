import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
from scipy import stats

img = cv.imread('echocardiography_speckle_noise.jpg')

#dst = cv.fastNlMeansDenoisingMulti(img, 2, 3, None, 4, 7, 21)
dst = cv.fastNlMeansDenoising(img, None, 21, 7, 21)

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def psnr(img, dst):
    mse = np.mean((img - dst) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX =255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

snr = signaltonoise(dst)
d = psnr(img, dst)
#snr = stats.signaltonoise(dst, axis= None)

plt.subplot(1,2,1), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2), plt.imshow(dst, cmap='gray')
plt.title('Filtering Image'), plt.xticks([]), plt.yticks([])

plt.show()