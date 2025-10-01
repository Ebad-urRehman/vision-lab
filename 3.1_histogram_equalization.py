# histogram equalization
# constrast limitive adaptive
# global

import cv2 as cv
# import matplotlib.pyplot as plt

image = cv.imread("captured_image.jpg", cv.IMREAD_GRAYSCALE)

cv.imshow("Actual Image", image)
cv.waitKey(0)

# global histogram equalization
equ = cv.equalizeHist(image)

cv.imshow("Global Equalized Image", image)

cv.waitKey(0)

# contrast limited adaptive histogram equalization
# title grid size : it diviedes all wall in equal parts Apply in parts
# cliplimit increase contrast intensity
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


# apply clahe function
clahe_image = clahe.apply(image)

cv.imshow("Local Adaptive Histogram Equalization", clahe_image)

cv.waitKey(0)


# plt.imshow(image.ravel(), 256, [0,256])

# making histogram has problem with Global HE

# plt.tightlayout