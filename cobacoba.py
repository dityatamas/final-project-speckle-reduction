from contrastenhancement import *
from RelaxedMedian import *
from despeckling import *
from SRAD import *
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from parameter import *
import random

#img = cv2.imread("D:\Tugas Akhir Bismillah\Echocardiography\plax sonosite\Frame 12 plax.jpg",0)
#img1 = cv2.imread("D:\Tugas Akhir Bismillah\Echocardiography\Apical_Four_Chamber\Frame 1 A4C.jpg",0)
img = cv2.imread("D:\Tugas Akhir Bismillah\Echocardiography\echocardiography_speckle_noise.jpg",0)
#img1 = cv2.imread()
#gauss = np.random.normal(0,0.7,img1.size)
#gauss = gauss.reshape(img1.shape).astype('uint8')
#speckle = img1 + img1 * gauss
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#noise = speckle-img1
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kontras = gammacorrection(img, 0.5)
f1 = rmf(kontras, 5, 9)
f2 = ad(kontras, 5, 10, 0.1, 1)#SRAD(kontras, 5, 0.1, 5, 1)
f3 = (f1*f2)
diff = img-f3#cv2.subtract(img,f3)

# get all non black Pixels
cntNotBlack = cv2.countNonZero(img)
cntNotBlack1 = cv2.countNonZero(f3)

# get pixel count of image
height, width = img.shape
print("Height : {0}\nWidth : {1}".format(height,width))
cntPixels = height*width

#snr1 = signaltonoise(img)
#print("SNR value is "+str(snr1))

# compute all black pixels
cntBlack = cntPixels - cntNotBlack
cntBlack1 = cntPixels - cntNotBlack1

plt.figure(figsize=(40, 30))
plt.subplot(231), plt.imshow(img,cmap='gray'), plt.xlabel("Number of black Pixels: {0}".format(cntBlack))
plt.xticks([]), plt.yticks([]), plt.title("Original Image")
plt.subplot(234), plt.hist(img.ravel(),256,[0,256]), plt.title("Noise Image")#, plt.xlabel("Pixel"), plt.ylabel("Number of Pixel")

plt.subplot(233), plt.imshow(diff,cmap='gray')
plt.xticks([]), plt.yticks([]), plt.title("Image with Speckle noise")
plt.subplot(236), plt.hist(diff.ravel(),  256,[0,256]), plt.xlabel("Pixel"), plt.ylabel("Number of Pixel")

plt.subplot(232), plt.imshow(f3,cmap='gray'), plt.xlabel("Number of black Pixels: {0}".format(cntBlack1))
plt.xticks([]), plt.yticks([]), plt.title("Filtering Image")
plt.subplot(235), plt.hist(f3.ravel(), 256,[0,256]), plt.title("HOSRAD Image")#, plt.xlabel("Pixel"), plt.ylabel("Number of Pixel")
"""
if img.ndim == 3:
    warnings.warn("Only grayscale images allowed.\n\tConverting to 2D Matrix...")
    img.mean(2)

#minpixel = np.amin(img)
#maxpixel = np.amax(img)
imgout = img.copy()
#imgout = imgout - minpixel / (maxpixel - minpixel)

deltaN = np.zeros_like(imgout)
deltaS = deltaN.copy()
deltaE = deltaN.copy()
deltaW = deltaN.copy()
g = np.ones_like(imgout)

M = imgout.shape[0]
N = imgout.shape[1]
niter = 10
gamma = 0.05
option = 1

for ii in range(niter):
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            deltaN = imgout[i, j - 1] - imgout[i, j]
            deltaS = imgout[i, j + 1] - imgout[i, j]
            deltaE = imgout[i + 1, j] - imgout[i, j]
            deltaW = imgout[i - 1, j] - imgout[i, j]

            # Normalized discrete gradient magnitude squared
            g2 = (deltaN ** 2 + deltaS ** 2 + deltaE ** 2 + deltaW ** 2) / imgout[i, j]

            # Normalized discrete laplacian
            l = (deltaN + deltaS + deltaE + deltaW) / imgout[i, j]

            # Speckle scale function
            num1 = np.std(imgout[i, j])
            den1 = np.mean(imgout[i, j])
            q0_squared = (num1 / den1) ** 2

            # Instantaneous coefficient of variation for edge detection
            num2 = (0.5 * g2) - ((1 / 16) * (l ** 2))
            den2 = (1 + ((1 / 4) * l)) ** 2
            q_squared = num2 / den2

            # conduction gradients
            c = (q_squared - q0_squared) / (q0_squared * (1 + q0_squared))
            if option == 1:
                g = 1 / (1 + c)
            elif option == 2:
                g = np.exp(-c)

            d = (g * deltaE) + (g * deltaW) + (g * deltaS) + (g * deltaN)
            imgout[i, j] = imgout[i, j] + (gamma / 4) * d

MSEawal = meansquareerror(img,0)
MSEbaru = meansquareerror(img,imgout)
PSNRawal = psnr(img,0)
PSNRbatu = psnr(img, imgout)
print("MSE awal dan baru : {0}".format(MSEawal)+" dan {0}".format(MSEbaru))
print("PSNR awal dan baru :{0}".format(PSNRawal)+" dan {0}".format(PSNRbatu))

plt.subplot(121),plt.imshow(imgout)
plt.subplot(122),plt.imshow(img) 
"""
plt.show()
