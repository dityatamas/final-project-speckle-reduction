def median_filter(src, filter_size):
    src = src.astype("float32")
    data = src.shape

    # Menentukan ukuran pixel yang dipakai, baik 3x3, 5x5 atau 7x7
    index = filter_size // 2
    data_final = src.copy()

    # Mengurutkan nilai pixel dari yang terkecil hingga terbesar
    for i in range(index, data[0] - index):
        for j in range(index, data[1] - index):
            temp = []
            for x in range(i - index, i + (index + 1)):
                for y in range(j - index, j + (index + 1)):
                    temp.append(src[x][y])
            # sort the values
            temp.sort()

            # Menentukan nilai tengah
            nilai_tengah = ((filter_size * filter_size) - 1) // 2
            med = temp[nilai_tengah]
            data_final.itemset((i, j), med)

    return data_final


def rmf(src, lower, upper):
    if src.ndim == 3:
        src.mean(2)

    med = median_filter(src, 3)

    lower_median = median_filter(src, lower)
    uppper_median = median_filter(src, upper)

    medd = src.copy()

    for i in range(1, src.shape[0] - 1):
        for j in range(1, src.shape[1] - 1):
            if (medd[i, j] != lower_median[i, j]) and (medd[i, j] != uppper_median[i, j]):
                medd[i, j] = med[i, j]

    imgout = medd
    imgout = imgout.astype("uint8")

    return imgout