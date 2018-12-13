import cv2
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import math

import pywt
import pywt.data

denoised_level = 3

def sgn(num):
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0

# Construct Gabor filter 
def build_filters():
     filters = []
     ksize = [7,9,11,13,15,17] # 6 gabor scales
     lamda = np.pi/2.0         # wave length
     for theta in np.arange(0, np.pi, np.pi / 4): #gabor direction，0°，45°，90°，135°，共四个
         for K in range(6):
             kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
             kern /= 1.5*kern.sum()
             filters.append(kern)
     plt.figure(1)

     # plot gabor filters
     for temp in range(len(filters)):
         plt.subplot(4, 6, temp + 1)
         plt.imshow(filters[temp])
     plt.show()
     return filters

# Gabor filtering process
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC1,kern)
        np.maximum(accum, fimg, accum)
    return accum

# Gabor features extraction
def getGabor(img,filters):
    res = [] # filter result
    for i in range(len(filters)):
        res1 = process(img, filters[i])
        res.append(np.asarray(res1))

    # Demonstrate filter result
    plt.figure(2)
    for temp in range(len(res)):
        plt.subplot(4,6,temp+1)
        plt.imshow(res[temp], cmap='gray' )
    plt.show()

    return res

def denoise(original_img,lev):
    # Wavelet transform of image
    coeffs2 = pywt.wavedec2(original_img, 'bior3.5', level = denoised_level)

    ###'a' parameter in eclectic function of hard and soft threshold 
    a = 0.5
    ##################Denoise#########################
    thcoeffs2 =[]
    for t in range(1, len(coeffs2)):
        tempcoeffs2 = []
        for i in range(0,3):
            tmp = coeffs2[t][i].copy()
            Sum = 0.0
            for j in coeffs2[t][i]:
                for x in j:
                    Sum = Sum + abs(x)
            N = coeffs2[t][i].size
            Sum = (1.0 / float(N)) * Sum
            sigma = (1.0 / 0.6745) * Sum
            lamda = sigma * math.sqrt(2.0 * math.log(float(N), math.e))
            for x in tmp:
                for y in x:
                    if(abs(y) >= lamda):
                        y = sgn(y) * (abs(y) - a * lamda)
                    else:
                        y = 0.0
            tempcoeffs2.append(tmp)
        thcoeffs2.append(tempcoeffs2)

    usecoeffs2 = []
    usecoeffs2.append(coeffs2[0])
    usecoeffs2.extend(thcoeffs2)

    #denoised_img correspond to denoised image
    denoised_img = pywt.waverec2(usecoeffs2, 'bior3.5')
    ##################Display#########################
    titles = ['Initial Image', ' Denoised Image']
    fig = plt.figure(figsize=(12, 3))
    #Display the  original image
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(original_img, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[0], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    #Display the denoised image
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(denoised_img, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[1], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.show()
    return denoised_img


def get_glgcm_features(mat):
    '''We base on Gray Level-Gradient Co-occurrence Matrix to calculate texture features,which includes small gradients dominance, big gradients dominance, gray level asymmetry, gradients asymmetry, energy, gray level mean, gradients mean,
        gray level variance, gradients variance, correlation, gray level entropy, gradients entropy, mixed entropy, inertia and inverse difference moment'''
    sum_mat = mat.sum()
    small_grads_dominance = big_grads_dominance = gray_asymmetry = grads_asymmetry = energy = gray_mean = grads_mean = 0
    gray_variance = grads_variance = corelation = gray_entropy = grads_entropy = entropy = inertia = differ_moment = 0
    sum_of_squares = 0
    for i in range(mat.shape[0]):
        gray_variance_temp = 0
        for j in range(mat.shape[1]):
            small_grads_dominance += mat[i][j] / ((j + 1) ** 2)
            big_grads_dominance += mat[i][j] * j ** 2
            energy += mat[i][j] ** 2
            if mat[i].sum() != 0:
                gray_entropy -= mat[i][j] * np.log(mat[i].sum())
            if mat[:, j].sum() != 0:
                grads_entropy -= mat[i][j] * np.log(mat[:, j].sum())
            if mat[i][j] != 0:
                entropy -= mat[i][j] * np.log(mat[i][j])
                inertia += (i - j) ** 2 * np.log(mat[i][j])
            differ_moment += mat[i][j] / (1 + (i - j) ** 2)
            gray_variance_temp += mat[i][j] ** 0.5

        gray_asymmetry += mat[i].sum() ** 2
        gray_mean += i * mat[i].sum() ** 2
        gray_variance += (i - gray_mean) ** 2 * gray_variance_temp
    for j in range(mat.shape[1]):
        grads_variance_temp = 0
        for i in range(mat.shape[0]):
            grads_variance_temp += mat[i][j] ** 0.5
        grads_asymmetry += mat[:, j].sum() ** 2
        grads_mean += j * mat[:, j].sum() ** 2
        grads_variance += (j - grads_mean) ** 2 * grads_variance_temp
    small_grads_dominance /= sum_mat
    big_grads_dominance /= sum_mat
    gray_asymmetry /= sum_mat
    grads_asymmetry /= sum_mat
    gray_variance = gray_variance ** 0.5
    grads_variance = grads_variance ** 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            corelation += (i - gray_mean) * (j - grads_mean) * mat[i][j]
    glgcm_features = [small_grads_dominance, big_grads_dominance, gray_asymmetry, grads_asymmetry, energy, gray_mean, grads_mean,
        gray_variance, grads_variance, corelation, gray_entropy, grads_entropy, entropy, inertia, differ_moment]
    return np.round(glgcm_features, 4)


def glgcm(original_img, ngrad=16, ngray=16):
    '''Gray Level-Gradient Co-occurrence Matrix,after normalization,set both gray level value and gradients value to 16'''
    img_gray = denoise(original_img,denoised_level)
    # utilize sobel operator to calculate gradients value on x-y directons each
    gsx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gsy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    height, width = img_gray.shape
    grad = (gsx ** 2 + gsy ** 2) ** 0.5 # Calculate gradients
    grad = np.asarray(1.0 * grad * (ngrad-1) / grad.max(), dtype=np.int16)
    gray = np.asarray(1.0 * img_gray * (ngray-1) / img_gray.max(), dtype=np.int16) # range 0-255 transformed into 0-15
    gray_grad = np.zeros([ngray, ngrad]) # Gray Level-Gradient Co-occurrence Matrix
    for i in range(height):
        for j in range(width):
            gray_value = gray[i][j]
            grad_value = grad[i][j]
            gray_grad[gray_value][grad_value] += 1
    gray_grad = 1.0 * gray_grad / (height * width) # Normalize gray level-gradient co-occurrence matrix to reduce the amount of calculation
    
    glgcm_features = get_glgcm_features(gray_grad)
    return list(glgcm_features)

def features(img):
    features = glgcm(img, ngrad=16, ngray=16)
    coeffs2 = pywt.wavedec2(img, 'coif5') # utilize 2D discrete wavelet transform to derive approximation and detail coefficients on every scales within the limits of maximum decomposition level
    #coeffs2 = getGabor(img, build_filters())
    energy_wav = []
    tmp = coeffs2[0].copy()
    Sum = 0
    tmp = 1.0 * tmp / (tmp.size)  # normalize coefficient array
    for x in tmp:
        for y in x:
            Sum += pow(y,2)    # calculate energy of each coefficient component
    energy_wav.append(Sum)
    for t in range(1, len(coeffs2)):
        for i in range(0,3):
            tmp = coeffs2[t][i].copy()
            tmp = 1.0 * tmp / (tmp.size)
            Sum = 0
            for x in tmp:
                for y in x:
                    Sum += pow(y,2)    # calculate energy of each coefficient component
            energy_wav.append(Sum)
    # select inverse difference moment,sum of squares of gray level-gradients co-occurence matrix and energy approximation after 2D DWT to form the finalized features vector
    vector = [features[4],features[14]]
    vector.extend(energy_wav)
    return vector
