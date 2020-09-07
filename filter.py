import numpy as np
import cv2
from matplotlib import pyplot as plt
from parameter import *
from medianfilter import *
from contrastenhancement import *

img = cv2.imread("D:\Tugas Akhir Bismillah\Echocardiography\echocardiography_speckle_noise.jpg", 0)
print(img.shape)
print(len(img))
print(len(img[0]))

"""
img2 = img[:,:,2]
print(img2.shape[0])
#a=0
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        if img2[i,j]>230:
            img2[i,j]=255
        else:
            img2[i,j]=0"""

lg = preprocessing_log(img, 0.06, 25)
print(lg)
#print(a)
#print(img[:,:,1])
#print(img[:,:,2])
#dst = cv2.fastNlMeansDenoisingMulti(img, 2, 3, None, 4, 7, 21)
dst5 = median_filter(lg, 5)
dst3 = median_filter(lg, 3)
dst7 = median_filter(lg, 7)
#print(img.shape)
#print(img2.shape)
#print(img2)

ms = meansquareerror(lg, dst5)
snr = signaltonoise(dst5)
d = psnr(lg, dst5)
print('Peak Signal to Noise Ratio :', d)
print('Mean Square Error          :', ms)
print('Signal to Noise Ratio      :', snr)
#plt.text()
#snr = stats.signaltonoise(dst, axis= None)

plt.subplot(2,2,1), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2), plt.imshow(dst3, cmap='gray')
plt.title('Filtering Image 3x3'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3), plt.imshow(dst5, cmap='gray')
plt.title('Filtering Image 5x5'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4), plt.imshow(lg, cmap='gray')
plt.title('Filtering Image 7x7'), plt.xticks([]), plt.yticks([])

plt.show()
