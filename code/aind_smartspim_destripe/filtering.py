"""
Script where the filtering algorithms are defined
"""

from typing import List, Optional, Tuple

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


def normalize_image(images: List[np.array]) -> np.ndarray:
    """
    Normalizes the images in a range between
    1.0 and 2.0

    Parameters
    ----------
    images: List[np.array]
        Images to normalize

    Returns
    -------
    np.ndarray
        Normalized image(s)
    """

    images = np.array(images)
    min_val = np.min(images)
    max_val = np.max(images)
    imgs_minus_min = images - min_val
    max_min = max_val - min_val
    normalized_imgs = 1 + np.divide(imgs_minus_min, max_min).astype(np.float16)

    return normalized_imgs


def invert_image(image: np.array) -> np.ndarray:
    """
    Inverts image signal

    Parameters
    ----------
    image: np.array
        Image data to invert

    Returns
    -------
    np.ndarray
        Inverted image
    """

    image = np.array(image)
    inv_image = image.max() - image
    return inv_image


def get_hemisphere_flatfield(
    image_path: str, tile_config: dict, flatfields: List[np.array]
) -> np.array:
    """
    Gets the hemisphere flatfield from the
    current laser.

    Parameters
    ----------
    image_path: str
        Path where the image data is located.

    tile_config: dict
        Tile configuration in terms of XY
        locations and brain side

    flatfields: List[np.array]
        List of flatfields applied per brain
        hemisphere. Usually, 2 per laser.

    Raises
    ------
    KeyError:
        Raises whenever we are trying to reach
        a tile that does not have a configuration
        brain side.

    Returns
    -------
    np.array
        Flatfield that will be used to correct
        the tiles from the corresponding hemisphere
    """

    splitted_image_path = str(image_path).split("/")
    XY_location_folders = splitted_image_path[-2].split("_")
    x_folder = XY_location_folders[0]
    y_folder = XY_location_folders[1]

    x_config = tile_config.get(x_folder)

    if x_config is None:
        raise KeyError(
            f"Please, check the tile config while trying to reach: {x_folder}"
        )

    brain_side = tile_config[x_folder].get(y_folder)

    if brain_side is None:
        raise KeyError(
            f"Please, check the tile config while trying to reach: {y_folder}"
        )

    return flatfields[brain_side]


def flatfield_correction(
    image_tiles: List[np.array],
    flatfield: np.array,
    darkfield: np.array,
    baseline: Optional[np.array] = None,
) -> np.array:
    """
    Corrects smartspim shadows in the tiles generated
    at the SmartSPIM light-sheet microscope.

    Parameters
    ----------
    image_tiles: List[np.array]
        Image tiles that will be corrected

    flatfield: np.array
        Estimated flatfield

    darkfield: np.array
        Estimated darkfield

    baseline: np.array
        Estimated baseline.
        Default: None

    Returns
    -------
    np.array
        Corrected tiles
    """

    image_tiles = np.array(image_tiles)

    if image_tiles.ndim != flatfield.ndim:
        flatfield = np.expand_dims(flatfield, axis=0)

    if image_tiles.ndim != darkfield.ndim:
        darkfield = np.expand_dims(darkfield, axis=0)

    darkfield = darkfield[: image_tiles.shape[-2], : image_tiles.shape[-1]]

    if darkfield.shape != image_tiles.shape:
        raise ValueError(
            f"Please, check the shape of the darkfield. Image shape: {image_tiles.shape} - Darkfield shape: {darkfield.shape}"
        )

    if flatfield.shape != image_tiles.shape:
        raise ValueError(
            f"Please, check the shape of the flatfield. Image shape: {image_tiles.shape} - Flatfield shape: {flatfield.shape}"
        )

    if baseline is None:
        baseline = np.zeros((image_tiles.shape[0],))

    baseline_indxs = tuple([slice(None)] + ([np.newaxis] * (image_tiles.ndim - 1)))

    # Subtracting dark field
    negative_darkfield = np.where(image_tiles <= darkfield)
    positive_darkfield = np.where(image_tiles > darkfield)

    # subtracting darkfield
    image_tiles[negative_darkfield] = 0
    image_tiles[positive_darkfield] = (
        image_tiles[positive_darkfield] - darkfield[positive_darkfield]
    )

    # Applying flatfield
    corrected_tiles = image_tiles / flatfield - baseline[baseline_indxs]

    # Converting back to uint16
    corrected_tiles = np.clip(corrected_tiles, 0, 65535).astype("uint16")

    return corrected_tiles


def filter_stripes(
    image: np.array,
    input_path: str,
    no_cells_config: dict,
    cells_config: dict,
    shadow_correction: Optional[dict] = None,
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

    input_path: str
        Path where the image data is located.

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

    # Filtering shadows if provided
    if shadow_correction is not None:
        retrospective = shadow_correction.get("retrospective")
        flatfield = shadow_correction.get("flatfield")
        darkfield = shadow_correction.get("darkfield")
        tile_config = shadow_correction.get("tile_config")

        # Get the corresponding flatfield from the prospective approach
        if not retrospective:
            flatfield = get_hemisphere_flatfield(
                image_path=input_path, tile_config=tile_config, flatfields=flatfield
            )

        filtered_image = flatfield_correction(
            image_tiles=filtered_image,
            flatfield=flatfield,
            darkfield=darkfield,
            baseline=None,
        )

    return filtered_image
