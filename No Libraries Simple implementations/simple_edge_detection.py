# libraries only for plotting
import numpy as np
import cv2 as cv


"""
# uncomment this to use simple 3x3 grayscale image
image = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
], dtype=np.uint8)
"""

image = cv.imread('image.png', cv.IMREAD_GRAYSCALE)

cv.imshow("Image", np.array(image))
cv.waitKey(0)

"""
What Kernel for edge detection does : Find difference between 
intensities of neigbouring pixels and thus making an edge for more different pixels
"""
# simplest horizontal kernel
kernel = np.array([-1, 1])

# output kernel matrix
output = []

rows = image.shape[0]
cols = image.shape[1]

# creating a matrix for output i.e. 3x2
output = np.zeros((rows, cols - 1), dtype=np.int32)

# goes through 3x3 grayscale image matrix in 1x2 window
for row in range(rows):
    for col in range(cols - 1):
        # applying convolution
        output[row, col] = image[row, col] * kernel[0] + image[row, col + 1] * kernel[1]

# output
output_display = np.clip(output, 0, 255).astype(np.uint8)

# Print result
for r in output:
    print(r)

# show edges image
cv.imshow("Edges Image", np.array(output_display))
cv.waitKey(0)