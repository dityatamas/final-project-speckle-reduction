import numpy as np
import cv2
import warnings

def srad1(img, niter, gamma, option):
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed.\n\tConverting to 2D Matrix...")
        img.mean(2)

    # minpixel = np.amin(img)
    # maxpixel = np.amax(img)
    imgout = img.copy()
    # imgout = imgout - minpixel / (maxpixel - minpixel)

    deltaN = np.zeros_like(imgout)
    deltaS = deltaN.copy()
    deltaE = deltaN.copy()
    deltaW = deltaN.copy()

    M = imgout.shape[0]
    N = imgout.shape[1]

    for ii in range(niter):
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                deltaN[i,j] = imgout[i, j - 1] - imgout[i, j]
                deltaS[i,j] = imgout[i, j + 1] - imgout[i, j]
                deltaE[i,j] = imgout[i + 1, j] - imgout[i, j]
                deltaW[i,j] = imgout[i - 1, j] - imgout[i, j]

                # Normalized discrete gradient magnitude squared
                g2 = (deltaN[i,j] ** 2 + deltaS[i,j] ** 2 + deltaE[i,j] ** 2 + deltaW[i,j] ** 2) / imgout[i, j]

                # Normalized discrete laplacian
                l = (deltaN[i,j] + deltaS[i,j] + deltaE[i,j] + deltaW[i,j]) / imgout[i, j]

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

                #d = (g * deltaE) + (g * deltaW) + (g * deltaS) + (g * deltaN)
                imgout[i, j] = imgout[i, j] + (gamma / 4) * g * (deltaN[i,j]+deltaS[i,j]+deltaE[i,j]+deltaW[i,j])

    return imgout