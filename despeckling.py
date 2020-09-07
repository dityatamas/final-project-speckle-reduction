import numpy as np
import warnings
import scipy.ndimage.filters as flt
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

def rgb2gray(rgb):

    #nilai piksel dari R = [:,:,0], G = [:,:,1], dan B = [:,:,2]
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2126*r + 0.7152*g + 0.0722*b

    return gray

def ad(img, niter, K, gamma, option):
    if img.ndim == 3:
        img = img.mean(2)
    img = img.astype('float32')
    imgout = img.copy()

    for ii in range(niter):
        for i in range(1, imgout.shape[0] - 1):
            for j in range(1, imgout.shape[1] - 1):
                deltaN = imgout[i, j - 1] - imgout[i, j]
                deltaS = imgout[i, j + 1] - imgout[i, j]
                deltaE = imgout[i + 1, j] - imgout[i, j]
                deltaW = imgout[i - 1, j] - imgout[i, j]

                if option == 1:
                    gN = np.exp(-(deltaN / K) ** 2.)
                    gS = np.exp(-(deltaS / K) ** 2.)
                    gE = np.exp(-(deltaE / K) ** 2.)
                    gW = np.exp(-(deltaW / K) ** 2.)
                elif option == 2:
                    gN = 1. / (1. + (deltaN / K) ** 2)
                    gS = 1. / (1. + (deltaS / K) ** 2)
                    gE = 1. / (1. + (deltaE / K) ** 2)
                    gW = 1. / (1. + (deltaW / K) ** 2)

                imgout[i, j] = imgout[i, j] * (1 - gamma * (gN + gS + gE + gW)) + (gamma * (gN * imgout[i, j - 1] + gS * imgout[i, j + 1] + gE * imgout[i + 1, j] + gW * imgout[i - 1, j]))

    return imgout


def srad1(img, niter, gamma, kernel, option):
    if img.ndim == 3:
        img = img.mean(2)

    img = img.astype('float32')
    imgout = img.copy()

    for ii in range(niter):
        for i in range(1, imgout.shape[0]-1):
            for j in range(1, imgout.shape[1]-1):
                dN = imgout[i,j-1] - imgout[i,j]
                dS = imgout[i,j+1] - imgout[i,j]
                dE = imgout[i+1,j] - imgout[i,j]
                dW = imgout[i-1,j] - imgout[i,j]

                #gm2 = dN**2 + dS**2 + dE**2 + dW**2/imgout[i,j]**2
                #laplacian = imgout[i,j+1] + imgout[i,j-1] + imgout[i+1,j] + imgout[i-1,j]/imgout[i,j]

                q2N = 0.5*(dN**2/imgout[i,j]**2) - 0.0625*((imgout[i,j-1]/imgout[i,j])**2)/(1+(0.25*(imgout[i,j-1]/imgout[i,j])))**2
                q2S = 0.5*(dS**2/imgout[i,j]**2) - 0.0625*((imgout[i,j+1]/imgout[i,j])**2)/(1+(0.25*(imgout[i,j+1]/imgout[i,j])))**2
                q2E = 0.5*(dE**2/imgout[i,j]**2) - 0.0625*((imgout[i+1,j]/imgout[i,j])**2)/(1+(0.25*(imgout[i+1,j]/imgout[i,j])))**2
                q2W = 0.5*(dW**2/imgout[i,j]**2) - 0.0625*((imgout[i-1,j]/imgout[i,j])**2)/(1+(0.25*(imgout[i-1,j]/imgout[i,j])))**2

                img_mean = uniform_filter(img, kernel)
                img_sqr_mean = uniform_filter(img ** 2, kernel)
                img_var = img_sqr_mean - img_mean ** 2
                overall_var = variance(img)
                q02 = img_var / (img_var + overall_var)

                if option == 1:
                    gN = np.exp(-(q2N - q02 / (q02 * (1 + q02))))
                    gS = np.exp(-(q2S - q02 / (q02 * (1 + q02))))
                    gE = np.exp(-(q2E - q02 / (q02 * (1 + q02))))
                    gW = np.exp(-(q2W - q02 / (q02 * (1 + q02))))
                elif option == 2:
                    gN = 1 / (1 + (q2N - q02 / (q02 * (1 + q02))))
                    gS = 1 / (1 + (q2S - q02 / (q02 * (1 + q02))))
                    gE = 1 / (1 + (q2E - q02 / (q02 * (1 + q02))))
                    gW = 1 / (1 + (q2W - q02 / (q02 * (1 + q02))))

                imgout[i, j] = imgout[i, j] * (1 - gamma * (gN + gS + gE + gW)) + (gamma * (gN * imgout[i, j - 1] + gS * imgout[i, j + 1] + gE * imgout[i + 1, j] + gW * imgout[i - 1, j]))
                #imgout[i,j] = imgout[i, j] + (gamma/4)*(gN*dN + gS*dS + gE*dE + gW*dW)
    return imgout

def ad1(img, niter, kappa, gamma, option, step=(1.,1.),sigma=0):
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for ii in np.arange(1, niter):

        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        if 0 < sigma:
            deltaSf = flt.gaussian_filter(deltaS, sigma);
            deltaEf = flt.gaussian_filter(deltaE, sigma);
        else:
            deltaSf = deltaS;
            deltaEf = deltaE;

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaSf / kappa) ** 2.) / step[0]
            gE = np.exp(-(deltaEf / kappa) ** 2.) / step[1]
        elif option == 2:
            gS = 1. / (1. + (deltaSf / kappa) ** 2.) / step[0]
            gE = 1. / (1. + (deltaEf / kappa) ** 2.) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

    return imgout


import numpy as np
import cv2
import warnings


def SRAD(src, niter, gamma, filter_size, option):
    img = src.copy()

    img = cv2.normalize(src.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point
    # M, N = img.shape

    imgout = img.copy()
    # imgout = np.exp(imgout)

    index = filter_size // 2
    data_final = imgout.copy()

    deltaN = np.zeros_like(imgout)
    deltaS = deltaN.copy()
    deltaE = deltaN.copy()
    deltaW = deltaN.copy()
    g2 = deltaN.copy()
    l = deltaN.copy()
    q0_squared = deltaN.copy()
    q_squared = deltaN.copy()
    g = deltaN.copy()

    M = imgout.shape[0]
    N = imgout.shape[1]

    for ii in range(niter):
        # Homogeneous ROI to calculate the ICOV
        for p in range(index, M - index):
            for q in range(index, N - index):
                q0_squared[p, q] = np.std(data_final[p, q]) / ((np.mean(data_final[p, q]))** 2)

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                deltaN[i, j] = imgout[i, j - 1] - imgout[i, j]
                deltaS[i, j] = imgout[i, j + 1] - imgout[i, j]
                deltaE[i, j] = imgout[i + 1, j] - imgout[i, j]
                deltaW[i, j] = imgout[i - 1, j] - imgout[i, j]

                # Normalized discrete gradient magnitude squared
                g2[i, j] = (deltaN[i, j] ** 2 + deltaS[i, j] ** 2 + deltaE[i, j] ** 2 + deltaW[i, j] ** 2) / ((imgout[i, j]) ** 2)

                # Normalized discrete laplacian
                l[i, j] = (deltaN[i, j] + deltaS[i, j] + deltaE[i, j] + deltaW[i, j]) / imgout[i, j]

                # Instantaneous coefficient of variation for edge detection
                q_squared[i, j] = ((0.5 * g2[i, j]) - ((1 / 16) * (l[i, j] ** 2))) / ((1 + ((1 / 4) * l[i, j])) ** 2)

                # conduction gradients
                if option == 1:
                    g[i, j] = 1 / (1 + (q_squared[i, j] - q0_squared[i, j]) / (q0_squared[i, j] * (1 + q0_squared[i, j])))
                elif option == 2:
                    g[i, j] = np.exp(-(q_squared[i, j] - q0_squared[i, j]) / (q0_squared[i, j] * (1 + q0_squared[i, j])))

                g[i, j] = np.nan_to_num(g[i, j])

                # d = (g * deltaE) + (g * deltaW) + (g * deltaS) + (g * deltaN)
                imgout[i, j] = imgout[i, j] + (gamma / 4) * (g[i, j] * deltaN[i, j] + g[i, j + 1] * deltaS[i, j] + g[i + 1, j] * deltaE[i, j] + g[i, j] *deltaW[i, j])

    # imgout = imgout.astype("uint8")

    return imgout

def OSRAD(src, niter, gamma, kernel, option):
    img = src.copy()

    img = cv2.normalize(src.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point
    # M, N = img.shape

    imgout = img.copy()
    # imgout = np.exp(imgout)

    index = kernel // 2
    data_final = imgout.copy()

    deltaN = np.zeros_like(imgout)
    deltaS = deltaN.copy()
    deltaE = deltaN.copy()
    deltaW = deltaN.copy()
    g2 = deltaN.copy()
    l = deltaN.copy()
    q0_squared = deltaN.copy()
    q_squared = deltaN.copy()
    g = deltaN.copy()
    abc = np.ones_like(imgout)
    ctang = 1

    M = imgout.shape[0]
    N = imgout.shape[1]

    for ii in range(niter):
        # Homogeneous ROI to calculate the ICOV
        for p in range(index, M - index):
            for q in range(index, N - index):
                q0_squared[p, q] = np.std(data_final[p, q]) / ((np.mean(data_final[p, q]))** 2)

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                deltaN[i, j] = imgout[i, j - 1] - imgout[i, j]
                deltaS[i, j] = imgout[i, j + 1] - imgout[i, j]
                deltaE[i, j] = imgout[i + 1, j] - imgout[i, j]
                deltaW[i, j] = imgout[i - 1, j] - imgout[i, j]

                # Normalized discrete gradient magnitude squared
                g2[i, j] = (deltaN[i, j] ** 2 + deltaS[i, j] ** 2 + deltaE[i, j] ** 2 + deltaW[i, j] ** 2) / ((imgout[i, j]) ** 2)

                # Normalized discrete laplacian
                l[i, j] = (deltaN[i, j] + deltaS[i, j] + deltaE[i, j] + deltaW[i, j]) / imgout[i, j]

                # Instantaneous coefficient of variation for edge detection
                q_squared[i, j] = ((0.5 * g2[i, j]) - ((1 / 16) * (l[i, j] ** 2))) / ((1 + ((1 / 4) * l[i, j])) ** 2)

                # conduction gradients
                if option == 1:
                    g[i, j] = 1 / (1 + (q_squared[i, j] - q0_squared[i, j]) / (q0_squared[i, j] * (1 + q0_squared[i, j])))
                elif option == 2:
                    g[i, j] = np.exp(-(q_squared[i, j] - q0_squared[i, j]) / (q0_squared[i, j] * (1 + q0_squared[i, j])))

                g[i, j] = np.nan_to_num(g[i, j])

                im = np.identity(2, dtype=float)
                ik = [(g[i,j]),ctang]
                ikS = [(g[i, j+1]), ctang]
                ikE = [(g[i+1, j]), ctang]
                d = ik*im
                dS = ikS*im
                dE = ikE*im

                abc[i,j] = abc[i,j] * (gamma / 4) * (d * deltaN[i, j] + dS * deltaS[i, j] + dE * deltaE[i, j] + d *deltaW[i, j])
                imgout[i, j] = imgout[i, j] + abc[i,j]
    # imgout = imgout.astype("uint8")

    return imgout