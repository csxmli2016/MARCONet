# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch

import numpy as np
import sys
sys.path.append('../')
from util import utils_image as util
# from . import utils_image as util

import random
from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth
import os
from PIL import Image
import math

from torch.nn import functional as F
from util.diffjpeg import DiffJPEG
from scipy import special
from torchvision.transforms.functional_tensor import rgb_to_grayscale
np.seterr(invalid='ignore')


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', 
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dirs):
    images = []
    assert os.path.isdir(dirs), '%s is not a valid directory' % dirs

    for root, _, fnames in sorted(os.walk(dirs)):
        fnames.sort()
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images



"""
# --------------------------------------------
# anisotropic Gaussian kernels
# --------------------------------------------
"""
def analytic_kernel(k):
    """Calculate the X4 kernel from the X2 kernel (for proof see appendix in paper)"""
    k_size = k.shape[0]
    # Calculate the big kernels size
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    # Loop over the small kernel to fill the big one
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
    # Crop the edges of the big kernel to ignore very small values and increase run time of SR
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    # Normalize to 1
    return cropped_big_k / cropped_big_k.sum()


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """ generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.

    Returns:
        k     : kernel
    """

    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x


def blur(x, k):
    '''
    x: image, NxcxHxW
    k: kernel, Nx1xhxw
    '''
    n, c = x.shape[:2]
    p1, p2 = (k.shape[-2]-1)//2, (k.shape[-1]-1)//2
    x = torch.nn.functional.pad(x, pad=(p1, p2, p1, p2), mode='replicate')
    k = k.repeat(1,c,1,1)
    k = k.view(-1, 1, k.shape[2], k.shape[3])
    x = x.view(1, -1, x.shape[2], x.shape[3])
    x = torch.nn.functional.conv2d(x, k, bias=None, stride=1, padding=0, groups=n*c)
    x = x.view(n, c, x.shape[2], x.shape[3])

    return x

def gen_kernel(k_size=np.array([15, 15]), scale_factor=np.array([4, 4]), min_var=0.6, max_var=10., noise_level=0):
    """"
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
    # max_var = 2.5 * sf
    """
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi  # random theta
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 - 0.5*(scale_factor - 1) # - 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0,1,3,2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    #raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

    # Normalize the kernel and return
    #kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    kernel = raw_kernel / np.sum(raw_kernel)
    return kernel


# --------------------------- util functions --------------------------- #
def sigma_matrix2(sig_x, sig_y, theta):
    """Calculate the rotated sigma matrix (two dimensional matrix).

    Args:
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.

    Returns:
        ndarray: Rotated sigma matrix.
    """
    D = np.array([[sig_x**2, 0], [0, sig_y**2]])
    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(U, np.dot(D, U.T))
def mesh_grid(kernel_size):
    """Generate the mesh grid, centering at zero.

    Args:
        kernel_size (int):

    Returns:
        xy (ndarray): with the shape (kernel_size, kernel_size, 2)
        xx (ndarray): with the shape (kernel_size, kernel_size)
        yy (ndarray): with the shape (kernel_size, kernel_size)
    """
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size,
                                                                           1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy


def pdf2(sigma_matrix, grid):
    """Calculate PDF of the bivariate Gaussian distribution.

    Args:
        sigma_matrix (ndarray): with the shape (2, 2)
        grid (ndarray): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size.

    Returns:
        kernel (ndarrray): un-normalized kernel.
    """
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel

def random_bivariate_generalized_Gaussian(kernel_size,
                                          sigma_x_range,
                                          sigma_y_range,
                                          rotation_range,
                                          beta_range,
                                          noise_range=None,
                                          isotropic=True):
    """Randomly generate bivariate generalized Gaussian kernels.

    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        beta_range (tuple): [0.5, 8]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    # assume beta_range[0] < 1 < beta_range[1]
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel = bivariate_generalized_Gaussian(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel


def random_generate_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0):
    scale = torch.rand(
        img.size(0), dtype=img.dtype, device=img.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    return generate_poisson_noise_pt(img, scale, gray_noise)


def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
    """Generate a bivariate isotropic or anisotropic Gaussian kernel.

    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (bool):

    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel
def random_bivariate_Gaussian(kernel_size,
                              sigma_x_range,
                              sigma_y_range,
                              rotation_range,
                              noise_range=None,
                              isotropic=True):
    """Randomly generate bivariate isotropic or anisotropic Gaussian kernels.

    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    kernel = bivariate_Gaussian(kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel

def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    """2D sinc filter, ref: https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter

    Args:
        cutoff (float): cutoff frequency in radians (pi is max)
        kernel_size (int): horizontal and vertical size, must be odd.
        pad_to (int): pad kernel size to desired size, must be odd or zero.
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)) / (2 * np.pi * np.sqrt(
                (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)), [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff**2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel

def random_mixed_kernels(kernel_list,
                         kernel_prob,
                         kernel_size=21,
                         sigma_x_range=[0.6, 5],
                         sigma_y_range=[0.6, 5],
                         rotation_range=[-math.pi, math.pi],
                         betag_range=[0.5, 8],
                         betap_range=[0.5, 8],
                         noise_range=None):
    """Randomly generate mixed kernels.

    Args:
        kernel_list (tuple): a list name of kenrel types,
            support ['iso', 'aniso', 'skew', 'generalized', 'plateau_iso',
            'plateau_aniso']
        kernel_prob (tuple): corresponding kernel probability for each
            kernel type
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        beta_range (tuple): [0.5, 8]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel (ndarray):
    """
    kernel_type = random.choices(kernel_list, kernel_prob)[0]
    if kernel_type == 'iso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range, isotropic=True)
    elif kernel_type == 'aniso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range, isotropic=False)
    elif kernel_type == 'generalized_iso':
        kernel = random_bivariate_generalized_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betag_range,
            noise_range=noise_range,
            isotropic=True)
    elif kernel_type == 'generalized_aniso':
        kernel = random_bivariate_generalized_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betag_range,
            noise_range=noise_range,
            isotropic=False)
    elif kernel_type == 'plateau_iso':
        kernel = random_bivariate_plateau(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, betap_range, noise_range=None, isotropic=True)
    elif kernel_type == 'plateau_aniso':
        kernel = random_bivariate_plateau(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, betap_range, noise_range=None, isotropic=False)
    return kernel

def random_add_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_poisson_noise_pt(img, scale_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out

"""
# --------------------------------------------
# Super-Resolution
# --------------------------------------------
#
# Kai Zhang (cskaizhang@gmail.com)
# https://github.com/cszn
# From 2019/03--2021/08
# --------------------------------------------
"""

def modcrop_np(img, sf):
    '''
    Args:
        img: numpy image, WxH or WxHxC
        sf: scale factor

    Return:
        cropped image
    '''
    w, h = img.shape[:2]
    im = np.copy(img)
    return im[:w - w % sf, :h - h % sf, ...]

def generate_poisson_noise_pt(img, scale=1.0, gray_noise=0):
    """Generate a batch of poisson noise (PyTorch version)

    Args:
        img (Tensor): Input image, shape (b, c, h, w), range [0, 1], float32.
        scale (float | Tensor): Noise scale. Number or Tensor with shape (b).
            Default: 1.0.
        gray_noise (float | Tensor): 0-1 number or Tensor with shape (b).
            0 for False, 1 for True. Default: 0.

    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    b, c, h, w = img.size()
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0
    if cal_gray_noise:
        img_gray = rgb_to_grayscale(img, num_output_channels=1)
        # round and clip image for counting vals correctly
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.
        # use for-loop to get the unique values for each sample
        vals_list = [len(torch.unique(img_gray[i, :, :, :].detach())) for i in range(b)]
        vals_list = [2**np.ceil(np.log2(vals)) for vals in vals_list]
        vals = img_gray.new_tensor(vals_list).view(b, 1, 1, 1)
        out = torch.poisson(img_gray * vals) / vals
        noise_gray = out - img_gray
        noise_gray = noise_gray.expand(b, 3, h, w)

    # alway calculate color noise
    # round and clip image for counting vals correctly
    img = torch.clamp((img * 255.0).round(), 0, 255) / 255.
    # use for-loop to get the unique values for each sample
    vals_list = [len(torch.unique(img[i, :, :, :].detach())) for i in range(b)]
    vals_list = [2**np.ceil(np.log2(vals)) for vals in vals_list]
    vals = img.new_tensor(vals_list).view(b, 1, 1, 1)
    out = torch.poisson(img * vals) / vals
    noise = out - img
    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    if not isinstance(scale, (float, int)):
        scale = scale.view(b, 1, 1, 1)
    return noise * scale

def random_bivariate_plateau(kernel_size,
                             sigma_x_range,
                             sigma_y_range,
                             rotation_range,
                             beta_range,
                             noise_range=None,
                             isotropic=True):
    """Randomly generate bivariate plateau kernels.

    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi/2, math.pi/2]
        beta_range (tuple): [1, 4]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    # TODO: this may be not proper
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel = bivariate_plateau(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)
    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)

    return kernel

def bivariate_generalized_Gaussian(kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True):
    """Generate a bivariate generalized Gaussian kernel.
        Described in `Parameter Estimation For Multivariate Generalized
        Gaussian Distributions`_
        by Pascal et. al (2013).

    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        beta (float): shape parameter, beta = 1 is the normal distribution.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None

    Returns:
        kernel (ndarray): normalized kernel.

    .. _Parameter Estimation For Multivariate Generalized Gaussian
    Distributions: https://arxiv.org/abs/1302.6498
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
    kernel = kernel / np.sum(kernel)
    return kernel

def bivariate_plateau(kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True):
    """Generate a plateau-like anisotropic kernel.
    1 / (1+x^(beta))

    Ref: https://stats.stackexchange.com/questions/203629/is-there-a-plateau-shaped-distribution

    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        beta (float): shape parameter, beta = 1 is the normal distribution.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None

    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.reciprocal(np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1)
    kernel = kernel / np.sum(kernel)
    return kernel


def generate_gaussian_noise_pt(img, sigma=10, gray_noise=0):
    """Add Gaussian noise (PyTorch version).

    Args:
        img (Tensor): Shape (b, c, h, w), range[0, 1], float32.
        scale (float | Tensor): Noise scale. Default: 1.0.

    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    b, c, h, w = img.size()
    if not isinstance(sigma, (float, int)):
        sigma = sigma.view(img.size(0), 1, 1, 1)
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0

    if cal_gray_noise:
        noise_gray = torch.randn(*img.size()[2:4], dtype=img.dtype, device=img.device) * sigma / 255.
        noise_gray = noise_gray.view(b, 1, h, w)

    # alway calculate color noise
    noise = torch.randn(*img.size(), dtype=img.dtype, device=img.device) * sigma / 255.

    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    return noise

def random_generate_gaussian_noise_pt(img, sigma_range=(0, 10), gray_prob=0):
    sigma = torch.rand(
        img.size(0), dtype=img.dtype, device=img.device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    return generate_gaussian_noise_pt(img, sigma, gray_noise)
def random_add_gaussian_noise_pt(img, sigma_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_gaussian_noise_pt(img, sigma_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out

def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h


def fspecial_laplacian(alpha):
    alpha = max([0, min([alpha,1])])
    h1 = alpha/(alpha+1)
    h2 = (1-alpha)/(alpha+1)
    h = [[h1, h2, h1], [h2, -4/(alpha+1), h2], [h1, h2, h1]]
    h = np.array(h)
    return h


def fspecial(filter_type, *args, **kwargs):
    '''
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    '''
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)
    if filter_type == 'laplacian':
        return fspecial_laplacian(*args, **kwargs)

"""
# --------------------------------------------
# degradation models
# --------------------------------------------
"""


def bicubic_degradation(x, sf=3):
    '''
    Args:
        x: HxWxC image, [0, 1]
        sf: down-scale factor

    Return:
        bicubicly downsampled LR image
    '''
    x = util.imresize_np(x, scale=1/sf)
    return x


def srmd_degradation(x, k, sf=3):
    ''' blur + bicubic downsampling

    Args:
        x: HxWxC image, [0, 1]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image

    Reference:
        @inproceedings{zhang2018learning,
          title={Learning a single convolutional super-resolution network for multiple degradations},
          author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
          booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
          pages={3262--3271},
          year={2018}
        }
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')  # 'nearest' | 'mirror'
    x = bicubic_degradation(x, sf=sf)
    return x


def dpsr_degradation(x, k, sf=3):

    ''' bicubic downsampling + blur

    Args:
        x: HxWxC image, [0, 1]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image

    Reference:
        @inproceedings{zhang2019deep,
          title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},
          author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
          booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
          pages={1671--1681},
          year={2019}
        }
    '''
    x = bicubic_degradation(x, sf=sf)
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    return x


def classical_degradation(x, k, sf=3):
    ''' blur + downsampling

    Args:
        x: HxWxC image, [0, 1]/[0, 255]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    #x = filters.correlate(x, np.expand_dims(np.flip(k), axis=2))
    st = 0
    return x[st::sf, st::sf, ...]


def add_sharpening(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening. borrowed from real-ESRGAN
    Input image: I; Blurry image: B.
    1. K = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * K + (1 - Mask) * I
    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    K = img + weight * residual
    K = np.clip(K, 0, 1)
    return soft_mask * K + (1 - soft_mask) * img


def add_blur(img, sf=4):
    wd2 = 4.0 + sf
    wd = 2.0 + 0.2*sf
    if random.random() < 0.3:
        l1 = wd2*random.random()
        l2 = wd2*random.random()
        k = anisotropic_Gaussian(ksize=2*random.randint(2,7)+3, theta=random.random()*np.pi, l1=l1, l2=l2)
    else:
        k = fspecial('gaussian', 2*random.randint(2,7)+3, wd*random.random())
    img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    return img


def add_resize(img, sf=4):
    rnum = np.random.rand()
    if rnum > 0.8:  # up
        sf1 = random.uniform(1, 2)
    elif rnum < 0.7:  # down
        sf1 = random.uniform(0.5/sf, 1)
    else:
        sf1 = 1.0
    img = cv2.resize(img, (int(sf1*img.shape[1]), int(sf1*img.shape[0])), interpolation=random.choice([1, 2, 3]))
    img = np.clip(img, 0.0, 1.0)

    return img


def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    if rnum > 0.6:   # add color Gaussian noise
        img += np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    elif rnum < 0.4: # add grayscale Gaussian noise
        img += np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:            # add  noise
        L = noise_level2/255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3,3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def add_speckle_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    img = np.clip(img, 0.0, 1.0)
    rnum = random.random()
    if rnum > 0.6:
        img += img*np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:
        img += img*np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = noise_level2/255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3,3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += img*np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def add_Poisson_noise(img):
    img = np.clip((img * 255.0).round(), 0, 255) / 255.
    vals = 10**(2*random.random()+2.0)  # [2, 4]
    if random.random() < 0.5:
        img = np.random.poisson(img * vals).astype(np.float32) / vals
    else:
        img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        img_gray = np.clip((img_gray * 255.0).round(), 0, 255) / 255.
        noise_gray = np.random.poisson(img_gray * vals).astype(np.float32) / vals - img_gray
        img += noise_gray[:, :, np.newaxis]
    img = np.clip(img, 0.0, 1.0)
    return img


def add_JPEG_noise(img):
    quality_factor = random.randint(30, 85) #random.randint(30, 95) 
    img = cv2.cvtColor(util.single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(util.uint2single(img), cv2.COLOR_BGR2RGB)
    return img


def random_crop(lq, hq, sf=4, lq_patchsize=64):
    h, w = lq.shape[:2]
    rnd_h = random.randint(0, h-lq_patchsize)
    rnd_w = random.randint(0, w-lq_patchsize)
    lq = lq[rnd_h:rnd_h + lq_patchsize, rnd_w:rnd_w + lq_patchsize, :]

    rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
    hq = hq[rnd_h_H:rnd_h_H + lq_patchsize*sf, rnd_w_H:rnd_w_H + lq_patchsize*sf, :]
    return lq, hq


def bsrgan_degradation(img, sf=4, lq_patchsize=72, isp_model=None):
    """
    This is the degradation model of BSRGAN from the paper
    "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"
    ----------
    img: HXWXC, [0, 1], its size should be large than (lq_patchsizexsf)x(lq_patchsizexsf)
    sf: scale factor
    isp_model: camera ISP model

    Returns
    -------
    img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
    hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
    """
    isp_prob, jpeg_prob, scale2_prob = 0.25, 0.8, 0.25#0.25, 0.9, 0.25
    # isp_prob, jpeg_prob, scale2_prob = 0.1, 0.5, 0.2#0.25, 0.9, 0.25
    sf_ori = sf
    h1, w1 = img.shape[:2]
    img = img.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]  # mod crop
    h, w = img.shape[:2]
    
    # if h < lq_patchsize*sf or w < lq_patchsize*sf:
    #     raise ValueError(f'img size ({h1}X{w1}) is too small!')

    hq = img.copy()

    if sf == 4 and random.random() < scale2_prob:   # downsample1
        if np.random.rand() < 0.5:
            img = cv2.resize(img, (int(1/2*img.shape[1]), int(1/2*img.shape[0])), interpolation=random.choice([1,2,3,4]))
        else:
            img = util.imresize_np(img, 1/2, True)
        img = np.clip(img, 0.0, 1.0)
        sf = 2
    
    shuffle_order = random.sample(range(7), 7)
    idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
    if idx1 > idx2:  # keep downsample3 last
        shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]

    for i in shuffle_order:
        if i == 0:
            img = add_blur(img, sf=sf)

        elif i == 1:
            img = add_blur(img, sf=sf)

        elif i == 2:
            a, b = img.shape[1], img.shape[0]
            # downsample2
            if random.random() < 0.75:
                sf1 = random.uniform(1,2*sf)
                img = cv2.resize(img, (int(1/sf1*img.shape[1]), int(1/sf1*img.shape[0])), interpolation=random.choice([1,2,3]))
            else:
                k = fspecial('gaussian', 15, random.uniform(0.1, 0.6*sf))
                k_shifted = shift_pixel(k, sf)
                k_shifted = k_shifted/k_shifted.sum()  # blur with shifted kernel
                img = ndimage.filters.convolve(img, np.expand_dims(k_shifted, axis=2), mode='mirror')
                img = img[0::sf, 0::sf, ...]  # nearest downsampling
            img = np.clip(img, 0.0, 1.0)

        elif i == 3:
            # downsample3
            img = cv2.resize(img, (int(1/sf*a), int(1/sf*b)), interpolation=random.choice([1,2,3]))
            img = np.clip(img, 0.0, 1.0)

        elif i == 4:
            # add Gaussian noise
            img = add_Gaussian_noise(img, noise_level1=3, noise_level2=25) #25 lxm

        elif i == 5:
            # add JPEG noise
            if random.random() < jpeg_prob:
                img = add_JPEG_noise(img)

        elif i == 6:
            # add processed camera sensor noise
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)

    # add final JPEG compression noise
    if random.random() > 0.6: #lxm
        img = add_JPEG_noise(img)

    # random crop
    # img, hq = random_crop(img, hq, sf_ori, lq_patchsize)

    return img, hq




def bsrgan_degradation_plus(img, sf=4, shuffle_prob=0.5, use_sharp=True, lq_patchsize=64, isp_model=None):
    """
    This is an extended degradation model by combining
    the degradation models of BSRGAN and Real-ESRGAN
    ----------
    img: HXWXC, [0, 1], its size should be large than (lq_patchsizexsf)x(lq_patchsizexsf)
    sf: scale factor
    use_shuffle: the degradation shuffle
    use_sharp: sharpening the img

    Returns
    -------
    img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
    hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
    """

    h1, w1 = img.shape[:2]
    img = img.copy()[:w1 - w1 % sf, :h1 - h1 % sf, ...]  # mod crop
    h, w = img.shape[:2]

    if h < lq_patchsize*sf or w < lq_patchsize*sf:
        raise ValueError(f'img size ({h1}X{w1}) is too small!')

    if use_sharp:
        img = add_sharpening(img)
    hq = img.copy()

    if random.random() < shuffle_prob:
        shuffle_order = random.sample(range(13), 13)
    else:
        shuffle_order = list(range(13))
        # local shuffle for noise, JPEG is always the last one
        shuffle_order[2:6] = random.sample(shuffle_order[2:6], len(range(2, 6)))
        shuffle_order[9:13] = random.sample(shuffle_order[9:13], len(range(9, 13)))

    poisson_prob, speckle_prob, isp_prob = 0.1, 0.1, 0.1

    for i in shuffle_order:
        if i == 0:
            img = add_blur(img, sf=sf)
        elif i == 1:
            img = add_resize(img, sf=sf)
        elif i == 2:
            img = add_Gaussian_noise(img, noise_level1=2, noise_level2=17) #25 lxm
        elif i == 3:
            if random.random() < poisson_prob:
                img = add_Poisson_noise(img)
        elif i == 4:
            if random.random() < speckle_prob:
                img = add_speckle_noise(img)
        elif i == 5:
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)
        elif i == 6:
            img = add_JPEG_noise(img)
        elif i == 7:
            img = add_blur(img, sf=sf)
        elif i == 8:
            img = add_resize(img, sf=sf)
        elif i == 9:
            img = add_Gaussian_noise(img, noise_level1=2, noise_level2=17) #25 lxm
        elif i == 10:
            if random.random() < poisson_prob:
                img = add_Poisson_noise(img)
        elif i == 11:
            if random.random() < speckle_prob:
                img = add_speckle_noise(img)
        elif i == 12:
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)
        else:
            print('check the shuffle!')

    # resize to desired size
    img = cv2.resize(img, (int(1/sf*hq.shape[1]), int(1/sf*hq.shape[0])), interpolation=random.choice([1, 2, 3]))

    # add final JPEG compression noise
    img = add_JPEG_noise(img)

    # random crop
    img, hq = random_crop(img, hq, sf, lq_patchsize)

    return img, hq

##############################################################################################################
#### Revised from Real-ESRGAN
##############################################################################################################
def tensor2im(input_image, norm=1, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    if norm == 1: #for clamp -1 to 1
        image_numpy = image_tensor[0].cpu().float().clamp_(0,1).numpy()
    elif norm == 2: # for norm through max-min
        image_ = image_tensor[0].cpu().float()
        max_ = torch.max(image_)
        min_ = torch.min(image_)
        image_numpy = (image_ - min_)/(max_-min_)*2-1
        image_numpy = image_numpy.numpy() 
    else:
        pass
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    # print(image_numpy.shape)
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    # print(image_numpy.shape)
    return image_numpy.astype(imtype)



def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kenrel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)

def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    # img_np = np.frombuffer(content, np.uint8)
    # img_np = content.astype(np.unit8)
    # imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    # img = cv2.imdecode(img_np, imread_flags[flag])
    # print(img_np)
    if float32:
        img = content.astype(np.float32) / 255.
    return img

def real_esrgan_degradation(gt, insf=4):
    # -------------------------------- Load gt images -------------------------------- #
    # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
    # gt_path = self.paths[index]
    # # avoid errors caused by high latency in reading files
    # retry = 3
    # while retry > 0:
    #     try:
    #         img_bytes = self.file_client.get(gt_path, 'gt')
    #     except Exception as e:
    #         logger = get_root_logger()
    #         logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
    #         # change another file to read
    #         index = random.randint(0, self.__len__())
    #         gt_path = self.paths[index]
    #         time.sleep(1)  # sleep 1s for occasional server congestion
    #     else:
    #         break
    #     finally:
    #         retry -= 1
    # img_gt = imfrombytes(img_bytes, float32=True)
    # img_gt = cv2.imread(gt_path)
    img_gt = gt
    kernel_range = [2 * v + 1 for v in range(3, 9)]

    blur_kernel_size = 13 #21 lxm
    kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    blur_sigma = [0.2, 3]
    betag_range = [0.5, 4]
    betap_range = [1, 2]
    sinc_prob = 0.1

    # blur settings for the second degradation
    blur_kernel_size2 = 21#opt['blur_kernel_size2']
    kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    blur_sigma2 = [0.2, 1.5]
    betag_range2 = [0.5, 4]
    betap_range2 = [1, 2]
    sinc_prob2 = 0.1

    final_sinc_prob = 0.4 #0.8 lxm 
    kernel_range = [2 * v + 1 for v in range(3, 9)]  # kernel size ranges from 7 to 21
    pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
    pulse_tensor[10, 10] = 1
    resize_prob = [0.2, 0.7, 0.1]
    resize_prob2 = [0.3, 0.4, 0.3]
    resize_range = [0.15, 1.5]
    resize_range2 = [0.3, 1.2]
    gray_noise_prob = 0.3 #0.4 lxm
    gray_noise_prob2 = 0.4
    gaussian_noise_prob = 0.3 #0.5 lxm
    gaussian_noise_prob2 = 0.3 #0.5 lxm
    noise_range = [1, 20] #[1, 30] lxm
    noise_range2 = [1, 15] #[1, 25]
    poisson_scale_range = [0.05, 3]
    poisson_scale_range2 = [0.05, 2.5]
    jpeg_range = [50, 95] #[30, 95] lxm
    jpeg_range2 = [50, 95] #[30, 95] lxm
    second_blur_prob = 0.3 #0.8 lxm
    scale = insf
    jpeger = DiffJPEG(differentiable=False)

    # -------------------- augmentation for training: flip, rotation -------------------- #
    # img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

    # crop or pad to 400: 400 is hard-coded. You may change it accordingly
    # h, w = img_gt.shape[0:2]
    # crop_pad_size = 400
    # # pad
    # if h < crop_pad_size or w < crop_pad_size:
    #     pad_h = max(0, crop_pad_size - h)
    #     pad_w = max(0, crop_pad_size - w)
    #     img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    # # crop
    # if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
    #     h, w = img_gt.shape[0:2]
    #     # randomly choose top and left coordinates
    #     top = random.randint(0, h - crop_pad_size)
    #     left = random.randint(0, w - crop_pad_size)
    #     img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

    # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < sinc_prob:
        # this sinc filter setting is for kernels ranging from [7, 21]
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            kernel_list,
            kernel_prob,
            kernel_size,
            blur_sigma,
            blur_sigma, [-math.pi, math.pi],
            betag_range,
            betap_range,
            noise_range=None)
    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < sinc_prob2:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel2 = random_mixed_kernels(
            kernel_list2,
            kernel_prob2,
            kernel_size,
            blur_sigma2,
            blur_sigma2, [-math.pi, math.pi],
            betag_range2,
            betap_range2,
            noise_range=None)

    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------------------- sinc kernel ------------------------------------- #
    if np.random.uniform() < final_sinc_prob:
        kernel_size = random.choice(kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel)
    else:
        sinc_kernel = pulse_tensor

    # BGR to RGB, HWC to CHW, numpy to tensor
    
    img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
    kernel = torch.FloatTensor(kernel)
    kernel2 = torch.FloatTensor(kernel2)

    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    gt = img_gt.unsqueeze(0).to(device)
    gt_usm = gt.to(device).clone()
    kernel1 = kernel.to(device)
    kernel2 = kernel2.to(device)
    sinc_kernel = sinc_kernel.to(device)

    ori_h, ori_w = gt.size()[2:4]
    out = gt_usm

    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(gt_usm, kernel1)
    # random resize
    updown_type = random.choices(['up', 'down', 'keep'], resize_prob)[0]
    if updown_type == 'up':
        scalelxm = np.random.uniform(1, resize_range[1])
    elif updown_type == 'down':
        scalelxm = np.random.uniform(resize_range[0], 1)
    else:
        scalelxm = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scalelxm, mode=mode)
    # noise

    if np.random.uniform() < gaussian_noise_prob:
        out = random_add_gaussian_noise_pt(
            out, sigma_range= noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=poisson_scale_range,
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
    out = torch.clamp(out, 0, 1)
    out = jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
    # blur
    if np.random.uniform() < second_blur_prob:
        out = filter2D(out, kernel2)
    # random resize
    updown_type = random.choices(['up', 'down', 'keep'], resize_prob2)[0]
    if updown_type == 'up':
        scalelxm = np.random.uniform(1, resize_range2[1])
    elif updown_type == 'down':
        scalelxm = np.random.uniform(resize_range2[0], 1)
    else:
        scalelxm = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
        out, size=(int(ori_h / scale * scalelxm), int(ori_w / scale * scalelxm)), mode=mode)
    # noise
    gray_noise_prob = gray_noise_prob2
    if np.random.uniform() < gaussian_noise_prob2:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=poisson_scale_range2,
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)

    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    if np.random.uniform() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // scale, ori_w // scale), mode=mode)
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // scale, ori_w // scale), mode=mode)
        out = filter2D(out, sinc_kernel)

    # clamp and round
    lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    return lq #0~1 tensor

if __name__ == '__main__':
    ImgPath = '/data/vdb/lxmF/RealD/TrainData/NWGPEN'
    SaveHQPath = '/data/vdb/lxmF/RealD/TrainData/SNE1014_2/HQ256'
    SaveLQPath = '/data/vdb/lxmF/RealD/TrainData/SNE1014_2/RealESRGAN'
    sf = 4

    if not os.path.exists(SaveHQPath):
        os.makedirs(SaveHQPath)
    if not os.path.exists(SaveLQPath):
        os.makedirs(SaveLQPath)


    # #################################################################
    # ##for my self
    # #################################################################
    # ImgPaths = make_dataset(ImgPath)
    # # random.shuffle(ImgPaths)
    # for i, ImgPath in enumerate(ImgPaths):
    #     ImgName = os.path.split(ImgPath)[-1]
        
    #     print([i, ImgName])

    #     img_gt = cv2.imread(ImgPath)
    #     img_gt = cv2.resize(img_gt, (256, 256))

    #     # ###Only DownSample
    #     s = random.randint(4,4)
    #     lq = cv2.resize(img_gt, (256//s, 256//s))

    #     ###Only Gaussian Blur
    #     # k = fspecial('gaussian', 2*random.randint(7,7)+3, 2.8*random.random())
    #     # lq = ndimage.filters.convolve(img_gt, np.expand_dims(k, axis=2), mode='mirror')

    #     # ###Only Noise
    #     # noise_level = random.randint(17, 17)
    #     # img = img_gt.astype(np.float32)
    #     # img += np.random.normal(0, noise_level, img_gt.shape).astype(np.float32)
    #     # lq = np.clip(img, 0.0, 255.0).astype(np.uint8)

    #     ###Only JPEG
    #     # img = img_gt
    #     # quality_factor = random.randint(30, 30)
    #     # result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    #     # lq = cv2.imdecode(encimg, 1)


    #     im = cv2.resize(lq, (256,256))
    #     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    #     im = Image.fromarray(im)
    #     im.save(os.path.join(SaveLQPath, ImgName))
    #     # exit('r')
    #     if i > 5000:
    #         break
    # #################################################################
    # ##for real esrgan
    # #################################################################
    ImgPaths = make_dataset(ImgPath)
    # random.shuffle(ImgPaths)
    for i, ImgPath in enumerate(ImgPaths):
        ImgName = os.path.split(ImgPath)[-1]
        
        print([i, ImgName])

        img_gt = cv2.imread(ImgPath)
        h, w = img_gt.shape[0:2]
        if random.random() > 1:
            lq = cv2.resize(img_gt, (h//4, w//4))
            im = cv2.resize(lq, (256,256), interpolation=0)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            # img_gt = cv2.resize(img_gt, (256,256))
            gt = imfrombytes(img_gt, float32=True)
            lq = real_esrgan_degradation(gt, random.choice([3,4,5,6]))
            im = tensor2im(lq)
            im =  cv2.resize(im, (256,256), interpolation=0)
        im = Image.fromarray(im)
        # im.save(os.path.join(SaveLQPath, ImgName))
        # exit('r')
        if i > 5000:
            break

    #################################################################
    ##for blind srgan
    #################################################################
    # ImgPaths = make_dataset(ImgPath)
    # # random.shuffle(ImgPaths)
    # for i, ImgPath in enumerate(ImgPaths):
    #     ImgName = os.path.split(ImgPath)[-1]
        
    #     print([i, ImgName])
    #     # Img = Image.open(ImgPath).convert('RGB')
    #     # Img = Img.resize((256,256), Image.BICUBIC)
    #     Img = util.imread_uint(ImgPath, 3)
    #     HQ = cv2.resize(Img, (256,256))
    #     h, w = Img.shape[0:2]
    #     if random.random() > 1:
    #         Img = cv2.resize(Img, (h//4, w//4))
    #         LQ = cv2.resize(Img, (256,256), interpolation=random.choice([1,2,3]))
    #     else:
    #         Img = cv2.resize(Img, (256,256))
    #         Img = util.uint2single(Img)
    #         LQ, _ = degradation_bsrgan(Img, sf=sf, lq_patchsize=256)
    #         LQ =  cv2.resize(util.single2uint(LQ), (int(sf*LQ.shape[1]), int(sf*LQ.shape[0])), interpolation=random.choice([1,2,3]))
    #     util.imsave(LQ, os.path.join(SaveLQPath, ImgName))
    #     util.imsave(HQ, os.path.join(SaveHQPath, ImgName))
    #     # exit('rr')
    #     if i > 5000:
    #         break


#     img = util.imread_uint('./test.png', 3)
#     img = util.uint2single(img)
    
    
#     for i in range(20):
#         img_lq, img_hq = degradation_bsrgan(img, sf=sf, lq_patchsize=72)
#         print(i)
#         lq_nearest =  cv2.resize(util.single2uint(img_lq), (int(sf*img_lq.shape[1]), int(sf*img_lq.shape[0])), interpolation=0)
#         img_concat = np.concatenate([lq_nearest, util.single2uint(img_hq)], axis=1)
#         util.imsave(img_concat, str(i)+'.png')

# #    for i in range(10):
# #        img_lq, img_hq = degradation_bsrgan_plus(img, sf=sf, shuffle_prob=0.1, use_sharp=True, lq_patchsize=64)
# #        print(i)
# #        lq_nearest =  cv2.resize(util.single2uint(img_lq), (int(sf*img_lq.shape[1]), int(sf*img_lq.shape[0])), interpolation=0)
# #        img_concat = np.concatenate([lq_nearest, util.single2uint(img_hq)], axis=1)
# #        util.imsave(img_concat, str(i)+'.png')

# #    run utils/utils_blindsr.py
