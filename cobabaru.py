import cv2
import os
from PIL import Image
from RelaxedMedian import *
from contrastenhancement import *
from despeckling import *
from timeit import default_timer as timer

#membuat video menjadi gambar per frame
cap = cv2.VideoCapture("D:\Tugas Akhir Bismillah\Echocardiography\plax_25fps.mp4")
start1 = timer()

try:
    #membuat folder
    if not os.path.exists('D:\Tugas Akhir Bismillah\Echocardiography\plax25fps_noise'):
        os.makedirs('D:\Tugas Akhir Bismillah\Echocardiography\plax25fps_noise')

#jika folder tidak terbuat maka akan timbul error
except OSError:
    print("Error: Creating directory data")

count = 1
#sec = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret==True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gauss = np.random.normal(0, 0.55, gray.size)
        gauss = gauss.reshape(gray.shape).astype('uint8')
        speckle = gray + gray * gauss
        #gc = gammacorrection(gray, 0.5)
        #rm = rmf(gc, 5, 9)
        #flt = SRAD(gc, 10, 0.1, 5, 1)
        #hybrid = rm*flt
        #cv2.imshow("Contoh", frame)
        cv2.imwrite("D:\\Tugas Akhir Bismillah\\Echocardiography\\plax25fps_noise\\Frame "+str(count)+" plax"+".jpg", speckle)

        print("{0} ".format(count)+"Reading {0} new frame".format(ret))
        count += 1
    else:
        break

akhir1 = timer() - start1
print("Processed time for the video to turn into images : {0} seconds".format(akhir1))

start2 = timer()
#menampilkan frame yang sudah disimpan menjadi gambar
print(os.getcwd())
os.chdir("D:\Tugas Akhir Bismillah\Echocardiography\plax25fps_noise")

meanh = 0
meanw = 0

numctr = len(os.listdir("."))

for file in os.listdir('.'):
    im = Image.open(os.path.join("D:\Tugas Akhir Bismillah\Echocardiography\plax25fps_noise", file))
    w, h = im.size
    meanw += w
    meanh += h
    #im.show()

meanh = int(meanh/numctr)
meanw = int(meanw/numctr)

print("The value of mean height : ", meanh)
print("The value of mean width : ", meanw)

#re-size citra agar memiliki panjang dan lebar yang sama
for file in os.listdir('.'):
    if file.endswith(".jpg"):
        # opening image using PIL Image
        im = Image.open(os.path.join("D:\Tugas Akhir Bismillah\Echocardiography\plax25fps_noise", file))

        # im.size includes the height and width of image
        w, h = im.size
        print(w, h)

        # resizing
        imResize = im.resize((meanw, meanh), Image.ANTIALIAS)
        imResize.save(file, 'JPEG', quality=95)  # setting quality
        # printing each resized image name
        print(im.filename.split('\\')[-1], " is resized")

citra = [img for img in os.listdir("D:\Tugas Akhir Bismillah\Echocardiography\plax25fps_noise")
         if img.endswith(".jpg")]

print(citra)

frames = cv2.imread(os.path.join("D:\Tugas Akhir Bismillah\Echocardiography\plax25fps_noise", citra[0]))

h, w, l = frames.shape
video = cv2.VideoWriter("plax_noise_25fps.avi", 0, 25, (w, h))
                      #(nama output file, belum tau, frame, shape/dimensi)

for ctr in citra:
    video.write(cv2.imread(os.path.join("D:\Tugas Akhir Bismillah\Echocardiography\plax25fps_noise", ctr)))

akhir2 = timer()-start2
print("Processed time for the images to turn into video : {0} seconds".format(akhir2))

vid = cv2.VideoCapture("D:\\Tugas Akhir Bismillah\\Echocardiography\\plax25fps_noise\\plax_noise_25fps.avi")

while(vid.isOpened()):
    rett, framee = vid.read()

    if rett==True :
        cv2.imshow("hasil", framee)
    else:
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
#video.release()
vid.release()
cv2.destroyAllWindows()
