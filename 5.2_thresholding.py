import cv2 as cv
import matplotlib.pyplot as plt


# load image in grayscale 

img = cv.imread("image.png", cv.IMREAD_GRAYSCALE)

_, global_thres = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

adaptive_thres = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11,2)

adaptive_guass = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11,2)

_, otsu_threshold = cv.threshold(img, 0, 255, cv.THRESH_OTSU, cv.THRESH_BINARY + cv.THRESH_OTSU)



titles = ['original', 'global thresholding', 'adaptive thresholding', 'adaptive gaussian', 'otsu threshold']

images = [img, global_thres, adaptive_thres, adaptive_guass, otsu_threshold]

plt.figure(figsize=(10,15))
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap="gray")
    # print(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.show()