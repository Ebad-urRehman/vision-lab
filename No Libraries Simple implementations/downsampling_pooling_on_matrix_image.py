import numpy as np
import matplotlib.pyplot as plt

# matplotlib config
plt.figure(figsize=(10,15))

# uncomment this to use simple 3x3 grayscale image
image = np.array([
    [10, 20, 30, 35],
    [40, 50, 60, 65],
    [70, 80, 90, 95],
    [100, 110, 120, 125]
], dtype=np.uint8)

# define kernel size 2x2
kernel_size = 2

"""
reshape(2,2,2,2) splits the 4×4 image into 4 mini-blocks of 2×2 pixels.
Axes mean dimensions :
here each axes(0,1,2,3) reprensent lenght of inner elements(2,2,2,2) i.e. 16 total elements arranged in 4D

max(axis=(1,3)) takes the max over the inner rows (1) and inner columns (3), leaving a 2×2 result that holds one pooled value per block.
Test it yourself to grasp
"""

# implementation of above
# it returned image in shape 2x2x2x2 total=16
reshaped_image = image.reshape(2, kernel_size, 2, kernel_size) 

# applying maximization by axis
max_pooled = reshaped_image.max(axis=(1,3))

print(max_pooled, max_pooled.shape)

# plotting
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title("Actual Matrix Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(max_pooled, cmap='gray')
plt.title("Max Pooled Matrix Image")
plt.axis('off')


plt.savefig("No Libraries Simple implementations/outputs/max_pooled_matrix.png", bbox_inches='tight', pad_inches=0.1)

plt.show()

plt.close()

