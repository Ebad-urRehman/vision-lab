import cv2 as cv
# import matplotlib.pyplot as plt

# image compression

image = cv.imread("image.png")
image_rbg = cv.cvtColor(image, cv.COLOR_BGR2RGB)

cv.imwrite("outputs/compressed_file.jpeg", image, [cv.IMWRITE_JPEG_QUALITY, 10])

com_jpg = cv.imread("outputs/compressed_file.jpeg")


com_jpg_rgb = cv.cvtColor(com_jpg, cv.COLOR_BGR2RGB)


cv.imshow('actual file', image_rbg)
cv.imshow('compressed file rgb', com_jpg_rgb)

cv.waitKey(0)


