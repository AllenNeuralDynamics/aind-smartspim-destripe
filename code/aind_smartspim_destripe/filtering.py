"""
Script where the filtering algorithms are defined
"""

import numpy as np
import pywt
from scipy import fftpack
from skimage import filters


def sigmoid(data: np.array):
    """
    Applies a sigmoid function

    Parameters
    -----------
    data: np.array
        Array with the nd data
    """
    return 1 / (1 + np.exp(-data))


def foreground_fraction(img: np.array, center: float, crossover: float):
    z = (img - center) / crossover
    f = sigmoid(z)
    return f


def get_foreground_background_mean(img, threshold_mask=0.3):
    cell_for = foreground_fraction(img.astype(np.float16), 400, 20)
    cell_for[cell_for > threshold_mask] = 1
    cell_for[cell_for <= threshold_mask] = 0

    foreground = img[cell_for == 1]
    background = img[cell_for == 0]

    foreground_mean = foreground.mean() if foreground.size else 0.0
    background_mean = background.mean() if background.size else 0.0

    return foreground_mean, background_mean, cell_for


def notch(n, sigma):
    """Generates a 1D gaussian notch filter `n` pixels long

    Parameters
    ----------
    n : int
        length of the gaussian notch filter
    sigma : float
        notch width

    Returns
    -------
    g : ndarray
        (n,) array containing the gaussian notch filter

    """
    if n <= 0:
        raise ValueError("n must be positive")
    else:
        n = int(n)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    x = np.arange(n)
    g = 1 - np.exp(-(x**2) / (2 * sigma**2))
    return g


def gaussian_filter(shape, sigma):
    """Create a gaussian notch filter

    Parameters
    ----------
    shape : tuple
        shape of the output filter
    sigma : float
        filter bandwidth

    Returns
    -------
    g : ndarray
        the impulse response of the gaussian notch filter

    """
    g = notch(n=shape[-1], sigma=sigma)
    g_mask = np.broadcast_to(g, shape).copy()
    return g_mask


def log_space_fft_filtering(
    input_image, wavelet="db3", level=0, sigma=64, max_threshold=4
):
    input_image_log = np.log(1.0 + input_image)
    coeffs = pywt.wavedec2(input_image_log, wavelet=wavelet, level=level)
    approx = coeffs[0]
    detail = coeffs[1:]

    width_fraction = sigma / input_image.shape[0]

    # print(f"Parameters: Wavelet {wavelet} coeffs: {coeffs} levels: {nb_levels}")
    # print(f"max threshold value: {max_threshold} sigma {sigma} width fraction {width_fraction}")

    coeff_filtered = [approx]
    for i, (ch, cv, cd) in enumerate(detail):
        ch_sq = ch**2
        ch_power = np.sqrt(ch_sq)

        otsu_threshold_sqrt = np.sqrt(
            filters.threshold_otsu(ch_sq)
        )  # threshold_otsu(ch_sq)
        threshold = min(max_threshold, otsu_threshold_sqrt)

        # print(f"Otsu threshold: {otsu_threshold_sqrt} - provided threshold {max_threshold}")
        # print(f"Selected threshold: {threshold}")

        mask = ch_power > threshold
        foreground = ch * mask
        background = ch * (1 - mask)

        background_means = np.broadcast_to(
            np.median(background, axis=-1)[:, np.newaxis], ch.shape
        )
        background_inpainted = background + background_means * mask

        fft = fftpack.rfft(background_inpainted, axis=-1)
        s = fft.shape[0] * width_fraction
        g = gaussian_filter(shape=fft.shape, sigma=s)
        background_filtered = fftpack.irfft(fft * g)

        ch_filtered = foreground + background_filtered * (1 - mask)

        coeff_filtered.append((ch_filtered, cv, cd))

    img_log_filtered = pywt.waverec2(coeff_filtered, wavelet)
    img_filtered = np.exp(img_log_filtered) + 1

    return img_filtered


def filter_steaks(image, no_cells_config, cells_config, microscope_high_int=2700):
    fore_mean, back_mean, cell_foreground_image = get_foreground_background_mean(image)
    filtered_image = None

    if fore_mean > back_mean and fore_mean > microscope_high_int:
        # It's an image with cells
        filtered_image = log_space_fft_filtering(input_image=image, **cells_config)
    else:
        # it's an image without cells
        filtered_image = log_space_fft_filtering(input_image=image, **no_cells_config)

    return filtered_image
