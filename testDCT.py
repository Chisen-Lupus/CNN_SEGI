from scipy.fft import fft, dct
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

image = plt.imread("experiments/sample_image/baboon-64-gray.png")[:,:,0]*2-1

imageDCT = dct(image, type=3, axis=1)
# if i==5 and j==5:
#     plt.imshow(imageGhostDCT,plt.cm.gray)
#     plt.show()
imageDCT = dct(imageDCT, type=3, axis=0)

plt.figure()
plt.subplot(2,5,1)
plt.title("image")
plt.imshow(image,plt.cm.gray)
plt.subplot(2,5,2)
plt.title("imageDCT")
plt.imshow(imageDCT,plt.cm.gray)
plt.show()