import numpy as np 
import cv2

from scipy.signal import correlate2d
from utils.util import generate_sobel
import os
import matlab.engine

if __name__ == "__main__":

    image = cv2.imread(os.getcwd()+r'\images\image_1.PNG')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    kernel = generate_sobel(7)

    eng = matlab.engine.start_matlab()

    cv2.imshow('grey', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    x = correlate2d(image,kernel,mode='same')
    np.savetxt('corr.txt',x)

    z = eng.importdata('corr.txt')
    eng.imshow(z)
