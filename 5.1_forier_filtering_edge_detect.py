# when image represented in frequencies (by forier transormation)
#  image -> frequency format 
# inverse_fourier(fequency_format) -> image

# two type of frequencies 
# low frequenies and high frequencies

# fourier removes low details of image



# we can apply two types of filters
# low pass filter, to extract fine details
# High pass filter, to extract obvious details


# img -> grayscale, frequncies, freq_shift

# 

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('image.png')

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

f = np.fft.fft2(img)

fshift = np.fft.fftshift(f)

rows, cols = img.shape

crow, ccol = rows // 2, cols // 2

# low pass filter

# radius
r=30

mask = np.zeros((rows, cols), np.uint8)  

mask[crow-r:crow+r, ccol-r:ccol+r] = 1

print(mask.shape)
print(mask)

# apply low pass

low_pass = fshift * mask

high_pass = fshift * (1-mask)


# inverse courier

low_image = np.abs(np.fft.fft2(np.fft.ifftshift(low_pass)))
high_image = np.abs(np.fft.fft2(np.fft.ifftshift(high_pass)))

plt.figure(figsize=(10,15))

plt.subplot(2,2,1)
plt.axis('off')
plt.title('Original Image')
plt.imshow(low_image)

plt.show()