import numpy as np
import ctypes as ct
import cv2


def twosComp (hex_str, num_bits=16):
    """
    Converts a hex string into a signed 16-bit integer
    
    @param hex_str: string containing the hexadecmimal value to be converted ex: '/FFAB'
    @param num_bits: int specifying the base of the result (default is 16)
    """

    value = int(hex_str,16)
    if value & (1 << (num_bits-1)):
        value -= (1<<num_bits)
    
    return value 
    
if __name__ == "__main__":

    garbage = []
    outputStreamFileName = 'fpgaOut.txt'
    
    expectedDim = 518
    expectedSize = expectedDim**2

    with open(outputStreamFileName,'r') as f:
        while True: 
            x = f.readline()
            y = f.readline()

            if not y: break

            # Remove the endline characters 
            x = x.rstrip('\n')
            y = y.rstrip('\n')
            
            #convert inputs to int 
            garbage_str = x+y
            garbage_str = garbage_str.replace(" ","")
            garbage_int = twosComp(garbage_str)

            garbage.append(garbage_int)


            # if (x_int > 127): garbage_int = ~garbage_int+1

    # Casting to uint8 for now, cv2 crashes with int16
    fpgaOut = np.asarray(garbage,dtype=np.uint8)
    buf = np.zeros(shape=[expectedSize],dtype=np.uint8)
    print(fpgaOut.shape)

    # Adjust fpgaOut shape based on fpgaOut size
    if (fpgaOut.shape[0] < expectedSize):
        buf[0:fpgaOut.shape[0]] = fpgaOut
    elif (fpgaOut.shape[0] > expectedSize):
        buf[0:expectedSize] = fpgaOut[0:expectedSize]
    else:
        buf = fpgaOut

    # Reshape into image sizes 
    buf = np.reshape(buf, [expectedDim,expectedDim])

    # Remove the zero padded layers from buf
    num_pad_layers = int(7/2)
    buf = buf[num_pad_layers:-num_pad_layers, num_pad_layers:-num_pad_layers]

    #Show output through OpenCV
    # buf_normalized = cv2.normalize(buf,dst=None,alpha=-32768,beta=32767,norm_type=cv2.NORM_MINMAX)

    cv2.imshow('Recieved Output',buf)
    cv2.waitKey(0)

    cv2.destroyAllWindows()