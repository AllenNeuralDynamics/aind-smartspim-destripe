"""
Script where the filtering algorithms are defined
"""

from typing import Optional, Tuple

import numpy as np
import pywt
from scipy import fftpack
from skimage import filters

from .types import PathLike


def sigmoid(data: np.array):
    """
    Applies a sigmoid function

    Parameters
    -----------
    data: np.array
        Array with the nd data
    """
    return 1 / (1 + np.exp(-data))


def foreground_fraction(img: np.array, center: float, crossover: float) -> float:
    """
    Gets the foreground fraction from the image
    using a sigmoid function.

    Parameters
    -----------
    img: np.array
        Image data

    center: float
        Intensity value considered to be
        the center of the image. Use statistical
        methods for this to get an overall value.

    crossover: float
        Crossover value to divide
        img - center

    Returns
    -----------
    float
        Foreground factor
    """
    z = (img - center) / crossover
    f = sigmoid(z)
    return f


def get_foreground_background_mean(
    img: np.array, threshold_mask: Optional[float] = 0.3
) -> Tuple:
    """
    Gets the foreground and background
    from an image. This needs to be improved
    since these values depend on current
    smartspim datasets.

    Parameters
    -----------
    img: np.array
        Image data

    threshold_mask: Optional[float]
        Threshold value used to
        get values as foreground or background.

    Returns
    -----------
    Tuple[float, float, np.array]
        Foreground mean, background mean and
        image mask looking for cells
    """
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
    input_image: np.array,
    wavelet: Optional[str] = "db3",
    level: Optional[int] = 0,
    sigma: Optional[int] = 64,
    max_threshold: Optional[int] = 4,
):
    """
    Filtering method to remove horizontal
    stripes (noise) from light-sheet SmartSPIM
    datasets using wavelet decomposition.

    The image processing steps are:
    1. Log function in the image to expand dark
    values.
    2. Wavelet decomposition.
    3. Get horizontal high frequency values from
    the decomposition.
    4. Mask the horizontal streaks using otsu
    thresholding.
    5. Remove the masked horizontal streaks.
    6. Compute a Real Fast-Fourier Transform.
    7. Gaussian filtering on the fft-d image to remove
    background stripes.
    8. Compute a reverse Real Fast-Fourier Transform.
    9. Inverse Discrete Wavelet Transform using
    the filtered horizonal coefficient but without
    modifying the vertical or diagonal coefficients.
    10. Reverse the log space from the fitlered image.

    Returns
    ----------
    np.array
        Image after removing horizontal
        streaks.
    """
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


def filter_steaks(
    image: np.array,
    no_cells_config: dict,
    cells_config: dict,
    microscope_high_int: Optional[int] = 2700,
) -> np.array:
    """
    Function to apply the desired fitlering
    function. At the moment, we only apply
    log space filtering on SmartSPIM datasets.

    Parameters
    -----------
    image: np.array
        Image data to be processed.

    no_cells_config: dict
        Dictionary with the parameters
        to clean SmartSPIM images without
        cells

    cells_config: dict
        Dictionary with the paramters
        to clean SmartSPIM images with
        cells

    microscope_high_int: Optional[it]
        High intensity output from the microscope.
        TODO: We need to improve the way how
        this number is calculated for every dataset

    Returns
    -----------
    np.array
        Filtered image data
    """
    fore_mean, back_mean, cell_foreground_image = get_foreground_background_mean(image)
    filtered_image = None

    if fore_mean > back_mean and fore_mean > microscope_high_int:
        # It's an image with cells
        filtered_image = log_space_fft_filtering(input_image=image, **cells_config)
    else:
        # it's an image without cells
        filtered_image = log_space_fft_filtering(input_image=image, **no_cells_config)

    return filtered_image
