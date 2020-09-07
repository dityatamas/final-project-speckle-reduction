import cv2
import numpy as np

def preprocessing_log(src, konstanta, nilaidaripixel) :
    RGB = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    images = RGB[:, :, 1]

    #semakin besar nilai C (konstanta), kecerahannya juga semakin cerah
    #semakin kecil nilai A (nilai dari pixel), kecerahannya semakin cerah
    g = konstanta * (np.log(nilaidaripixel+np.float64(images)))

    return g*255

def log_transform(src):
    gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    gray = gray(src)
    max_ = np.max(gray)
    return (255/np.log(1+max_)) * np.log(1+gray)


def gammacorrection(src, gamma):
    #nilai dari gamma (G) akan mempengaruhi kecerahan dari citra
    #G>1 akan membuat citra menjadi lebih gelap
    #G<1 akan membuat citra menjadi lebih cerah
    #G=1 tidak akan memberikan efek apapun
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in range(0, 256)]).astype("uint8")

    return cv2.LUT(src, table) 
