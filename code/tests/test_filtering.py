"""Tests for the smartspim filtering"""

import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.append("../")
from aind_smartspim_destripe import filtering


class SmartspimFiltering(unittest.TestCase):
    """Class for testing smartspim filtering"""

    def test_sigmoid(self):
        """Test the sigmoid function"""
        # Test with scalar
        self.assertAlmostEqual(filtering.sigmoid(np.array(0)), 0.5)
        self.assertAlmostEqual(filtering.sigmoid(np.array(-1)), 1 / (1 + np.exp(1)))
        self.assertAlmostEqual(filtering.sigmoid(np.array(1)), 1 / (1 + np.exp(-1)))

        # Test with array
        data = np.array([-1, 0, 1])
        expected = 1 / (1 + np.exp(-data))
        np.testing.assert_array_almost_equal(filtering.sigmoid(data), expected)

    def test_foreground_fraction(self):
        """Testing foreground fraction"""
        # Test with simple data
        img = np.array([10, 20, 30, 40, 50])
        center = 30
        crossover = 10

        z = (img - center) / crossover
        expected = 1 / (1 + np.exp(-z))
        np.testing.assert_array_almost_equal(
            filtering.foreground_fraction(img, center, crossover), expected
        )

    def test_get_foreground_background_mean(self):
        """Testing get foreground vs background mean"""
        # Test with simple data
        img = np.array([10, 20, 400, 500, 600])
        threshold_mask = 0.3

        # Compute expected results
        cell_for = filtering.foreground_fraction(img.astype(np.float16), 400, 20)
        cell_for[cell_for > threshold_mask] = 1
        cell_for[cell_for <= threshold_mask] = 0

        foreground = img[cell_for == 1]
        background = img[cell_for == 0]

        foreground_mean = foreground.mean() if foreground.size else 0.0
        background_mean = background.mean() if background.size else 0.0

        # Call the function
        fg_mean, bg_mean, mask = filtering.get_foreground_background_mean(
            img, threshold_mask
        )

        # Validate results
        self.assertAlmostEqual(fg_mean, foreground_mean)
        self.assertAlmostEqual(bg_mean, background_mean)
        np.testing.assert_array_equal(mask, cell_for)

    def test_empty_image_get_foreground_background_mean(self):
        """
        Testing get foreground vs background
        mean when the image is empty
        """
        # Test with an empty image
        img = np.array([])
        threshold_mask = 0.3

        fg_mean, bg_mean, mask = filtering.get_foreground_background_mean(
            img, threshold_mask
        )

        self.assertEqual(fg_mean, 0.0)
        self.assertEqual(bg_mean, 0.0)
        np.testing.assert_array_equal(mask, img)

    def test_no_foreground(self):
        """
        Testing when there is no foreground
        """
        # Test with all background values
        img = np.array([10, 20, 30, 40, 50])
        threshold_mask = 1.0  # No values will be above this threshold

        fg_mean, bg_mean, mask = filtering.get_foreground_background_mean(
            img, threshold_mask
        )

        self.assertEqual(fg_mean, 0.0)  # No foreground
        self.assertEqual(bg_mean, img.mean())  # All values are background
        np.testing.assert_array_equal(mask, np.zeros_like(img))

    def test_no_background(self):
        """
        Testing with no background in the image
        """
        img = np.array([400, 420, 430, 440, 460])
        threshold_mask = 0.0

        fg_mean, bg_mean, mask = filtering.get_foreground_background_mean(
            img, threshold_mask
        )

        self.assertEqual(fg_mean, img.mean())
        self.assertEqual(bg_mean, 0.0)
        np.testing.assert_array_equal(mask, np.ones_like(img))

    def test_notch(self):
        """testing notch function"""
        # Test with valid inputs
        n = 5
        sigma = 1.0
        result = filtering.notch(n, sigma)
        expected = 1 - np.exp(-(np.arange(n) ** 2) / (2 * sigma**2))
        np.testing.assert_array_almost_equal(result, expected)

        # Test with n = 1 (edge case)
        self.assertAlmostEqual(filtering.notch(1, sigma)[0], 1 - np.exp(0))

        with self.assertRaises(ValueError):
            filtering.notch(0, sigma)  # n <= 0
        with self.assertRaises(ValueError):
            filtering.notch(-1, sigma)  # n <= 0
        with self.assertRaises(ValueError):
            filtering.notch(n, -1)  # sigma <= 0

    def test_gaussian_filter(self):
        """
        Testing gaussian filter
        """
        shape = (3, 5)
        sigma = 1.0
        result = filtering.gaussian_filter(shape, sigma)
        expected_notch = filtering.notch(shape[-1], sigma)
        expected = np.broadcast_to(expected_notch, shape)
        np.testing.assert_array_almost_equal(result, expected)

        # Test with edge case: shape = (1, 1)
        shape = (1, 1)
        result = filtering.gaussian_filter(shape, sigma)
        np.testing.assert_array_equal(result, np.array([[1 - np.exp(0)]]))

    def test_log_space_fft_filtering(self):
        """
        Testing stripe removal with synthetic horizontal stripes
        """
        # Create a synthetic image with horizontal stripes
        input_image = np.tile(np.linspace(1, 100, 100), (100, 1)).astype(np.float32)
        wavelet = "db3"
        level = 1
        sigma = 64
        max_threshold = 4

        # Apply the filter
        result = filtering.log_space_fft_filtering(
            input_image, wavelet, level, sigma, max_threshold
        )

        # Validate the result
        self.assertEqual(result.shape, input_image.shape)
        self.assertTrue(np.all(result > 0))  # Ensure no negative values in the result

    def test_log_space_fft_filtering_small_image(self):
        """
        Testing filtering with a very small image
        """
        # Edge case: small image
        input_image = np.random.rand(4, 4).astype(np.float32)
        result = filtering.log_space_fft_filtering(
            input_image, wavelet="db3", level=1, sigma=64, max_threshold=4
        )
        self.assertEqual(result.shape, input_image.shape)

    def test_normalize_image(self):
        """
        Testing image normalization
        """
        images = [np.array([[1, 2], [3, 4]]), np.array([[0, 5], [10, 15]])]
        normalized = filtering.normalize_image(images)
        self.assertGreaterEqual(normalized.min(), 1.0, "Minimum value should be >= 1.0")
        self.assertLessEqual(normalized.max(), 2.0, "Maximum value should be <= 2.0")
        self.assertEqual(
            normalized.shape, (2, 2, 2), "Output shape should match input list shape"
        )

    def test_invert_image(self):
        """
        Testing image invert
        """
        image = np.array([[0, 1], [2, 3]])
        inverted = filtering.invert_image(image)
        expected = np.array([[3, 2], [1, 0]])
        np.testing.assert_array_equal(
            inverted, expected, "Inverted image values incorrect"
        )

    def test_get_hemisphere_flatfield(self):
        """
        Testing the function tog et the flatfields that
        come from the SmartSPIM microscope. The microscope
        has two lasers and there is one flat per laser.
        """
        tile_config = {"X1": {"Y1": 0, "Y2": 1}, "X2": {"Y1": 0, "Y2": 1}}
        flatfields = [np.array([[1, 1], [1, 1]]), np.array([[2, 2], [2, 2]])]
        flatfield = filtering.get_hemisphere_flatfield("X1_Y1", tile_config, flatfields)
        np.testing.assert_array_equal(
            flatfield, flatfields[0], "Incorrect flatfield returned"
        )

        flatfield = filtering.get_hemisphere_flatfield("X2_Y2", tile_config, flatfields)
        np.testing.assert_array_equal(
            flatfield, flatfields[1], "Incorrect flatfield returned"
        )

        with self.assertRaises(KeyError):
            filtering.get_hemisphere_flatfield("X3_Y1", tile_config, flatfields)

    def test_flatfield_correction(self):
        """
        Testing flatfield correction
        """
        image_tiles = np.array([[[10, 20], [30, 40]]])
        flatfield = np.array([[[2, 2], [2, 2]]])
        darkfield = np.array([[[1, 1], [1, 1]]])
        corrected = filtering.flatfield_correction(image_tiles, flatfield, darkfield)
        expected = np.array([[[4, 9], [14, 19]]], dtype=np.uint16)
        np.testing.assert_array_equal(
            corrected, expected, "Flatfield correction incorrect"
        )

        with self.assertRaises(ValueError):
            filtering.flatfield_correction(image_tiles, flatfield, darkfield[:-1])

    @patch("aind_smartspim_destripe.filtering.log_space_fft_filtering")
    @patch("aind_smartspim_destripe.filtering.get_foreground_background_mean")
    def test_filter_stripes(
        self, mock_get_foreground_background_mean, mock_log_space_fft_filtering
    ):
        """
        Tests filtering stripes
        """
        image = np.array([[10, 20], [30, 40]])
        no_cells_config = {"wavelet": "db3", "sigma": 64}
        cells_config = {"wavelet": "db3", "sigma": 64}

        mock_get_foreground_background_mean.return_value = (50, 5, None)
        mock_log_space_fft_filtering.return_value = image

        filtered_image = filtering.filter_stripes(
            image,
            "path/to/tile",
            no_cells_config,
            cells_config,
            shadow_correction=None,
        )
        np.testing.assert_array_equal(
            filtered_image, image, "Filtering output mismatch"
        )

        shadow_correction = {
            "retrospective": True,
            "flatfield": np.array([[2, 2], [2, 2]]),
            "darkfield": np.array([[1, 1], [1, 1]]),
            "tile_config": {},
        }
        filtered_image = filtering.filter_stripes(
            image,
            "path/to/tile",
            no_cells_config,
            cells_config,
            shadow_correction=shadow_correction,
        )
        self.assertIsNotNone(filtered_image, "Shadow correction not applied correctly")
