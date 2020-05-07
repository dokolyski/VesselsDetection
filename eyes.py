from skimage.exposure import equalize_adapthist, equalize_hist
from skimage.io import imread, imshow, show
from skimage.filters import meijering, sato, frangi, hessian, gaussian, unsharp_mask
from skimage.feature import canny
import skimage
import matplotlib.pyplot as plt
import numpy as np
from ml import accuracy, sensitivity, specificity

from sklearn.metrics import accuracy_score

def RMSE(matrix1, matrix2):
    sum = 0

    for y in range(matrix1.shape[0]):
        for x in range(matrix1.shape[1]):
            sum += (matrix1[y][x] - matrix2[y][x]) ** 2

    return sum ** 0.5

def normalize(image):
    out = image.astype(float)

    for row in out:
        row = row.astype(float)

    maxValue = np.max(image)
    if maxValue > 1:
        for row in range(out.shape[0]):
            for col in range(out.shape[1]):
                out[row][col] = float(out[row][col]) / 255.0
    return out

def showComparison(image1, image2):
    plt.subplot(121)
    imshow(image1, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.subplot(122)
    imshow(image2, cmap=plt.cm.gray, vmin=0, vmax=1)

    show()

def binarize(image):
    out = image.copy()

    threshold = np.percentile(image, 80)

    for row in range(out.shape[0]):
        for col in range(out.shape[1]):
            out[row][col] = 1 if out[row][col] > threshold else 0

    return out

def preprocess(image):
    next = image
    # next = image ** 2
    # showComparison(image, next)
    # image = next

    # next = equalize_hist(image)
    # showComparison(image, next)
    # image = next
    #
    next = equalize_adapthist(image)
    #showComparison(image, next)
    image = next

    # next = gaussian(image, sigma=2)
    # showComparison(image, next)
    # image = next
    #
    # next = unsharp_mask(image, radius=5, amount=2)
    # showComparison(image, next)
    # image = next

    return image

def process(image):
    # next = meijering(image, sigmas=[1])
    # #showComparison(image, next)
    # image = next

    # next = sato(image, sigmas=[1])
    # showComparison(image, next)
    # image = next

    next = frangi(image, sigmas=[1])
    next /= np.max(next)
    #showComparison(image, next)
    image = next

    next = binarize(image) #0.003
    # showComparison(image, next)
    image = next

    # next = hessian(image, sigmas=[1])
    # showComparison(image, next)
    # image = next

    return image

def routine(number):
    image = normalize(imread("eyes/healthy/" + number + ".jpg", as_gray=True))
    manual = normalize(imread("eyes/healthy_manual/" + number + ".jpg", as_gray=True))

    image = preprocess(image)
    image = process(image)

    manual = binarize(manual) #0.1
    print(RMSE(manual, image))
    print(accuracy(manual, image))
    showComparison(manual, image)

def main():
    for i in range(1,15 + 1):
        number = str(i)
        while len(number) < 3:
            number = "0" + number
        routine(number)
if __name__ == "__main__":
    main()