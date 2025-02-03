"""
Test readers
"""

import sys
import unittest

import numpy as np

sys.path.append("../")

from unittest.mock import patch

from aind_smartspim_destripe.readers import _get_extension, imread, raw_imread


class TestImageReader(unittest.TestCase):

    def test_get_extension(self):
        self.assertEqual(_get_extension("image.tif"), ".tif")
        self.assertEqual(_get_extension("/path/to/image.png"), ".png")
        self.assertEqual(_get_extension("C:\\Images\\image.raw"), ".raw")
        self.assertEqual(_get_extension("no_extension"), "")

    @patch("numpy.memmap")
    def test_raw_imread_big_endian(self, mock_memmap):
        """Test raw_imread with big-endian data."""
        mock_memmap.side_effect = [
            np.array([300, 200], dtype=">u4"),  # Big-endian header
            np.array([100, 50], dtype="<u4"),  # Little-endian header
            np.ones((300, 200), dtype=">u2"),  # Image data
        ]

        result = raw_imread("fake_path.raw")
        self.assertEqual(result.shape, (300, 200))
        self.assertEqual(result.dtype, np.dtype(">u2"))

    @patch("numpy.memmap")
    def test_raw_imread_little_endian(self, mock_memmap):
        """Test raw_imread with little-endian data."""
        mock_memmap.side_effect = [
            np.array([100, 50], dtype=">u4"),  # Big-endian header
            np.array([300, 200], dtype="<u4"),  # Little-endian header
            np.ones((300, 200), dtype="<u2"),  # Image data
        ]

        result = raw_imread("fake_path.raw")
        self.assertEqual(result.shape, (300, 200))
        self.assertEqual(result.dtype, np.dtype("<u2"))

    @patch("numpy.memmap")
    def test_raw_imread_invalid_path(self, mock_memmap):
        """Test raw_imread with an invalid path that raises an exception."""
        mock_memmap.side_effect = OSError("File not found")

        with self.assertRaises(OSError):
            raw_imread("invalid_path.raw")

    @patch("aind_smartspim_destripe.readers.raw_imread")
    @patch("tifffile.imread")
    @patch("imageio.imread")
    def test_imread(self, mock_iio_imread, mock_tifffile_imread, mock_raw_imread):
        mock_raw_imread.return_value = np.zeros((10, 10))
        mock_tifffile_imread.return_value = np.ones((10, 10))
        mock_iio_imread.return_value = np.full((10, 10), 255)

        self.assertTrue(np.array_equal(imread("image.raw"), np.zeros((10, 10))))
        self.assertTrue(np.array_equal(imread("image.tif"), np.ones((10, 10))))
        self.assertTrue(np.array_equal(imread("image.png"), np.full((10, 10), 255)))
