import os 
import matlab.engine
import numpy as np
from scipy.signal import correlate2d
import cv2  

from utils.util import *


def runFPGATest(image_path, kernel_size, kernel_type='sobel'):
    """
    Testing flow: 
        - Load image and preprocess via colorspace conversion and resizing
        - Zero pad image based on kernel 
        - generate kernel (either Sobel or identity)
        - quantize to fixed-point and write to file via MATLAB
        - Compile and run C# program for FPGA communication
        - Load output bytes file from C# and compute ground truth convolution
        - Compute norms of inputs and outputs 
      

    """

    # Process image and generate kernel
    greyscale = image_rescale(image_path,512)
    greyscale = zeroPad(greyscale,kernel_size)
    sobel = generate_sobel(kernel_size)

    # quantize through matlab
    fp_image, fp_kernel = quantize(greyscale,sobel)

    # generate bytefile for C#
    createUARTStream(fp_image,fp_kernel,"uart_input_bytes.txt")

    # compile and run C#
    runFPGAConvolution()

    # call helper method for convolution verification and exit to main instance
    return checkFPGAOutputs(greyscale,sobel,fp_kernel)


if __name__ == "__main__":

    dir = ""
    pathsList = os.listdir(dir)
    errors = []

    for path in pathsList:
        errors.append(runFPGATest(path,512))






