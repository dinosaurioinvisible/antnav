import cv2 as cv
import numpy as np
from source.pyDTW import DTW



def resize(shape):
    '''
    Return a function to resize image to given size
    '''
    return lambda im: cv.resize(im, shape, interpolation=cv.INTER_NEAREST)


def gauss_blur(kernel_shape=(3, 3)):
    '''
    Return a function to blur image
    given the kernel size and the mean
    '''
    return lambda im: cv.GaussianBlur(im, kernel_shape, 0)


def canny(upper, lower):
    '''
    Return a function to perform Canny edge detection
    within the given thresholds
    '''
    return lambda im: cv.Canny(cv.normalize(src=im, dst=im, aplha=0, ebta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U), upper, lower)


def standardize():
    return lambda im: (im - np.mean(im)) / np.std(im)


def scale021():
    #return lambda im: cv.normalize(src=im, dst=im, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    return lambda im: im.astype(np.float32)/np.max(im)


def wavelet(image_shape):
    dtw = DTW(image_shape)
    return lambda im: dtw.dwt_haar(im)[-1]


def mod_dtype(dtype):
    return lambda im: im.astype(dtype)


def lin(img, kernel_shape=(5, 5)):
    '''
    Local Image Normalisation
    Normalises and each pixel using 
    the mean of the neighboring pixels
    '''
    img = img.astype(np.int16, copy=False)
    mu = cv.blur(img, kernel_shape)
    img = img - mu
    return cv.normalize(src=img, dst=img, aplha=0, beta=255, norm_type=cv.NORM_MINMAX)
    # var = cv.blur(img*img, kernel_shape)
    # sig = var**0.5 + np.finfo(float).eps
    # return img / sig


def loc_norm(kernel_shape=(3, 3)):
    return lambda im: lin(im, kernel_shape)


def glin(img, sig1=2, sig2=20):
    '''
    Gaussian Local Image Normalisation
    Normalises and standarises each pixel using 
    the weighted mean the std.dev of the neighboring pixels
    The weighted mean is calculated using the gaussian kernel
    '''
    img = img.astype(np.int16, copy=False)
    mu = cv.GaussianBlur(img, (0, 0), sig1)
    img = img - mu
    var = cv.GaussianBlur(img*img, (0, 0), sig2)
    sig = var**0.5 + np.finfo(float).eps
    img = img / sig
    return cv.normalize(src=img, dst=img, aplha=0, ebta=255, norm_type=cv.NORM_MINMAX)


def gauss_loc_norm(sig1=2, sig2=20):
    return lambda im: glin(im, sig1, sig2)


def imvcrop(shape=None, vcrop=None):
    if vcrop < 1. and vcrop > 0:
        vcrop = int(round(shape[1] * vcrop))
    else:
        vcrop = None
    return lambda im: im[vcrop:, :]


def histeq():
    return lambda im: cv.equalizeHist(im)


def _quant(im, k=3):
    return np.floor(np.floor(im/(im.max()/k)) * (255/k)).astype(np.uint8)

def quant(k=3):
    return lambda im: _quant(im, k=k)


def make_pipeline(sets):
    '''
    Create a pre-processing pipeline from a dictionary of settings
    :param sets:
    :return:
    '''
    pipe = []
    if sets.get('shape'):
        pipe.append(resize(sets['shape']))
    if sets.get('vcrop'):
        im_shape = sets.get('shape')
        pipe.append(imvcrop(shape=im_shape, vcrop=sets.get('vcrop')))
    if sets.get('histeq'):
        pipe.append(histeq())
    if sets.get('blur'):
        pipe.append(gauss_blur())
    if sets.get('quant'):
        pipe.append(quant(k=sets.get('quant')))
    if sets.get('loc_norm'):
        pipe.append(loc_norm(**sets.get('loc_norm')))
    if sets.get('gauss_loc_norm'):
        pipe.append(gauss_loc_norm(**sets.get('gauss_loc_norm')))
    if sets.get('edge_range'):
        lims = sets['edge_range']
        pipe.append(canny(lims[0], lims[1])) 
    # range and data type processing
    if sets.get('normstd'):
        pipe.append(standardize())
    if sets.get('scale021'):
        pipe.append(scale021())
    if sets.get('type'):
        pipe.append(mod_dtype(sets.get('type')))
    else:
        # Always change the datatype to float32 to avoid wrap-around!!
        pipe.append(mod_dtype(np.float32))
    # if sets.get('wave'):
    #     im_size = sets.get('shape')
    #     pipe.append(wavelet(im_size))

    return pipe


class Pipeline:
    def __init__(self, **sets) -> None:
        if sets:
            self.pipe = make_pipeline(sets)
        else:
            self.pipe = []
            self.pipe.append(mod_dtype(np.float32))

    def apply(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        for p in self.pipe:
            imgs = [p(img) for img in imgs]
        return imgs if len(imgs) > 1 else imgs[0]