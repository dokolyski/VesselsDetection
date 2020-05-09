from skimage import img_as_uint
from skimage.color import rgb2gray
from skimage.exposure import equalize_adapthist, equalize_hist
from skimage.io import imread, imshow, show, imsave
from skimage.filters import meijering, sato, frangi, hessian, gaussian, unsharp_mask
from skimage.morphology import binary_erosion, disk, binary_closing, binary_dilation, label
from skimage.feature import canny
import skimage
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rescale, resize
from os import listdir
from skimage.restoration import denoise_nl_means, estimate_sigma

from skimage.filters import  threshold_local


from ml import accuracy, sensitivity, specificity


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


def binarize(image, threshold=-1):
    out = image.copy()

    if threshold == -1:
        threshold = threshold_local(out, 35, offset=0)
        for row in range(out.shape[0]):
            for col in range(out.shape[1]):
                out[row][col] = 1 if out[row][col] > threshold[row][col] else 0
        return out
    else:
        for row in range(out.shape[0]):
            for col in range(out.shape[1]):
                out[row][col] = 1 if out[row][col] > threshold else 0

        return out

def getImportanceMask(image):
    out = image

    threshold = 0.01

    mask = np.ndarray(shape=out.shape)
    for row in range(out.shape[0]):
        for col in range(out.shape[1]):
            if out[row][col] < threshold:
                mask[row][col] = 0
            else:
                mask[row][col] = 1
    return mask

def denoise(img):
    estimatedSigma = np.mean(estimate_sigma(img, multichannel=False))
    return denoise_nl_means(img, h=0.8 * estimatedSigma, fast_mode=True)

def preprocess(image):
    next = image

    next = equalize_adapthist(image)
    # showComparison(image, next)
    image = next

    return image


def process(image):
    next = frangi(image, sigmas=[1])
    next /= np.max(next)
    next **= 0.1
    # showComparison(image, next)
    image = next

    next = binarize(image, np.percentile(image, 80)) #0.003
    # showComparison(image, next)
    image = next

    return image


def resize_and_normalize(image, binary=False):
    max_width = 1000
    result_image = image

    if result_image.shape[1] > max_width:
        result_image = resize(result_image, (int(result_image.shape[0] * max_width / result_image.shape[1]), max_width), anti_aliasing=True)

    maxValue = np.max(image)
    if maxValue > 1:
        result_image = result_image / 255.0

    if binary:
        return binarize(result_image, 0.3)
    else:
        return result_image


def processSimple(image, manual):
    # to gray scale
    image = rgb2gray(image)

    # process
    importanceMask = getImportanceMask(image)
    erodedImportanceMask = binary_erosion(importanceMask, selem=disk(2))
    image = preprocess(image)
    image = process(image) * erodedImportanceMask

    denoisedImage = image

    denoisedImage = skimage.morphology.remove_small_objects(image > 0)

    showComparison(manual, denoisedImage)

    #importanceMask = binary_erosion(importanceMask)
    #print(RMSE(manual, image))
    print('IMAGE: ' + str(accuracy(manual, image)))
    print('WITH MASK: ' + str(accuracy(manual, image, importanceMask)))
    print('DENOISED: ' + str(accuracy(manual, denoisedImage, importanceMask)))
    print('\n')

    return denoisedImage

def colorizeVessels(rgb_image, mask):
    out = rgb_image.copy()
    for row in range(out.shape[0]):
        for col in range(out.shape[1]):
            if mask[row][col] == 1:
                out[row][col] = [0,1,0]
    return out

def main():
    # read all filenames in images directory
    images_directory = 'images/input'
    manual_directory = 'images/manual'
    images_filenames = listdir(images_directory)
    manual_filenames = listdir(manual_directory)
    images_filenames.sort()
    manual_filenames.sort()

    # iterate by files
    for i, m in zip(images_filenames, manual_filenames):
        image = resize_and_normalize((imread(images_directory + '/' + i)))
        manual = resize_and_normalize(imread(manual_directory + '/' + m, as_gray=True), True)

        vessels = processSimple(image, manual)
        colorized = colorizeVessels(image, vessels)
        showComparison(image, colorized)

if __name__ == "__main__":
    main()
