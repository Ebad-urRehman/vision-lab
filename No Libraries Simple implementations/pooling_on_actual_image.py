import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# matplotlib config
plt.figure(figsize=(10,15))

image = cv.imread('image.png', cv.IMREAD_GRAYSCALE)


"""
For a real image we have to first the pixels divisible by kernel size
so that they fit in pooling layer
"""


"""
Axes mean dimensions :
here each axes(0,1,2,3) reprensent lenght of inner elements(x,x,x,x)

max(axis=(1,3)) takes the max over the inner rows (1) and inner columns (3), leaving a 2×2 result that holds one pooled value per block.
Test it yourself to grasp
Check pooling_on_matrix_image.py file for basic understanding
"""

height, width = image.shape[:2]

# ------------------------------2x2 kernel size image--------------------------
# define kernel size 2x2
kernel_size = 2

height = height - (height % kernel_size)
width = width - (width % kernel_size)

# implementation of above
reshaped_image = image.reshape(height // kernel_size, kernel_size,  width // kernel_size, kernel_size) 

# applying maximization by axis
max_pooled2x2 = reshaped_image.max(axis=(1,3))

print(max_pooled2x2, max_pooled2x2.shape)

# ------------------------------2x2 kernel size image--------------------------
# define kernel size 3x3
kernel_size = 3

height = height - (height % kernel_size)
width = width - (width % kernel_size)

# implementation of above
reshaped_image = image.reshape(height // kernel_size, kernel_size,  width // kernel_size, kernel_size) 

# applying maximization by axis
max_pooled3x3 = reshaped_image.max(axis=(1,3))

print(max_pooled3x3, max_pooled3x3.shape)


# plotting
plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title(f"Actual Image\n  Shape : {image.shape[0]} * {image.shape[1]}")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(max_pooled2x2, cmap='gray')
plt.title(f"Max Pooled Image 2x2\n Shape : {max_pooled2x2.shape[0]} * {max_pooled2x2.shape[0]}")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(max_pooled3x3, cmap='gray')
plt.title(f"Max Pooled Image 3x3\n Shape : {max_pooled3x3.shape[0]} * {max_pooled3x3.shape[0]}")
plt.axis('off')


plt.savefig("No Libraries Simple implementations/outputs/max_pooled_image.png", bbox_inches='tight', pad_inches=0.1)

plt.show()

plt.close()

