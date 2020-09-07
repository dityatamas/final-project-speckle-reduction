import numba
from matplotlib import pyplot as plt
import cv2
from skimage import color

#@numba.jit(nopython=True)
def median_filter(src, filter_size):
    data = src.shape

    #Menentukan ukuran pixel yang dipakai, baik 3x3, 5x5 atau 7x7
    index = filter_size // 2
    data_final = src.copy()

    #Mengurutkan nilai pixel dari yang terkecil hingga terbesar
    for i in range(index, data[0] - index):
        for j in range(index, data[1] - index):
            temp = []
            for x in range(i - index, i + (index + 1)):
                for y in range(j - index, j + (index + 1)):
                    temp.append(src[x][y])
            #sort the values
            temp.sort()

            #Menentukan nilai tengah
            nilai_tengah = ((filter_size*filter_size)-1) // 2
            med = temp[nilai_tengah]
            data_final.itemset((i,j), med)

    return data_final 


