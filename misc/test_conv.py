import cv2 
import numpy as np 
import os
from scipy.signal import correlate2d
from utils.util import generate_sobel,zeroPad 
import ctypes as ct

def manual_conv(img,kernel):

    img_hgt = img.shape[0]
    img_wid = img.shape[1]

    kernel_hgt = kernel.shape[0]
    kernel_wid = kernel.shape[1]

    pixel_out_val = 0

    im_window_x = 0
    im_window_y = 0

    output = np.zeros(shape=[512,512],dtype=np.int16)

    for i in range(int(kernel_hgt/2),img_hgt-int(kernel_hgt/2)):

        for j in range(int(kernel_wid/2),img_wid-int(kernel_wid/2)):
            pixel_out_val = 0

            for kern_y in range(0,kernel_hgt):
                for kern_x in range(0,kernel_wid):

                    im_window_x = kern_x + (j - int(kernel_wid/2))
                    im_window_y = kern_y + (i - int(kernel_hgt/2))

                    pixel_out_val += np.int8(kernel[kern_x,kern_y]) * np.uint8(img[im_window_x,im_window_y])

            im_window_x = j - int(kernel_wid/2)
            im_window_y = i - int(kernel_hgt/2)
            output[im_window_x,im_window_y] = np.int16(pixel_out_val)
    
    return output


img_path = os.getcwd() + r"\images\image_4.PNG"
img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(512,512),interpolation=cv2.INTER_AREA)

kernel = generate_sobel(7)
padded = zeroPad(img, kernel.shape[0])
print(type(padded))

true_conv = correlate2d(img, kernel, 'same')
test_conv = manual_conv(padded,kernel)

np.savetxt('blah.txt',test_conv)
print(np.allclose(true_conv,test_conv,rtol=1e-5))