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

"""Horizontal edge detection"""
# simplest horizontal kernel
kernel = np.array([-1, 1])

# output kernel matrix
output = []

rows = image.shape[0]
cols = image.shape[1]

# creating a matrix for output i.e. 3x2
output_horizontal = np.zeros((rows, cols - 1), dtype=np.int32)

# goes through 3x3 grayscale image matrix in 1x2 window
for row in range(rows):
    for col in range(cols - 1):
        # applying convolution
        output_horizontal[row, col] = image[row, col] * kernel[0] + image[row, col + 1] * kernel[1]

# output
output_display = np.clip(output_horizontal, 0, 255).astype(np.uint8)

# using np.abs
# output_display = np.clip(np.abs(output_horizontal), 0, 255).astype(np.uint8) # more accurate but noisy edges


# show edges image
cv.imshow("Edges Image Horizontal", np.array(output_display))
cv.waitKey(0)

cv.imwrite("No Libraries Simple implementations/outputs/simple_horizontal_edge_detection.png", output_display)

# -------------------------------------              ---------------------------------

"""Vertical edge detection"""
# simplest vertical kernel
kernel = np.array([[-1], [1]])

# output kernel matrix
output = []

rows = image.shape[0]
cols = image.shape[1]

# creating a matrix for output i.e. 3x2
output_vertical = np.zeros((rows-1, cols), dtype=np.int32)

# goes through 3x3 grayscale image matrix in 1x2 window
for row in range(rows - 1):
    for col in range(cols):
        # applying convolution
        output_vertical[row, col] = image[row, col] * kernel[0] + image[row + 1, col] * kernel[1]

# output
output_display = np.clip(output_vertical, 0, 255).astype(np.uint8)

# more accuracte + noisy edges for this task
# output_display = np.clip(np.abs(output_vertical), 0, 255).astype(np.uint8)

# show edges image
cv.imshow("Edges Image Vertical", np.array(output_display))
cv.waitKey(0)

cv.imwrite("No Libraries Simple implementations/outputs/simple_vertical_edge_detection.png", output_display)


#----------------------------------------Both combined---------------------

# shapes of output_hor and ouput_ver are diff 
# making them same size
min_rows = min(output_horizontal.shape[0], output_vertical.shape[0])
min_cols = min(output_horizontal.shape[1], output_vertical.shape[1])

output_horizontal = output_horizontal[:min_rows, :min_cols]
output_vertical = output_vertical[:min_rows, :min_cols]

output = np.abs(output_horizontal) + np.abs(output_vertical)

# output
output_display = np.clip(output, 0, 255).astype(np.uint8)

# show edges image
cv.imshow("Edges Image Added", np.array(output_display))
cv.waitKey(0)

cv.imwrite("No Libraries Simple implementations/outputs/simple_hor_ver_added_edge_detection.png", output_display)

