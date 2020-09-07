import numpy as np
import random

def speckle(img, prob) :
    hasil = np.zeros(img.shape, np.uint8)
    threshold = 1-prob
    for i in range(img.shape[0]) :
        for j in range(img.shape[1]) :
            acak = random.random()
            if acak<prob :
                hasil[i][j] = 128
                for k in range(5) :
                    hasil[i-k][j-k] = 128+10*acak
            else :
                hasil[i][j] = img[i][j]
    return hasil