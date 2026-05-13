"""Tests for flatfield estimation"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.append("../")
from aind_smartspim_destripe.flatfield_estimation import (shading_correction,
                                                          slide_flat_estimation,
                                                          unify_fields)


class TestShadingCorrectionFunctions(unittest.TestCase):

    @patch("aind_smartspim_destripe.flatfield_estimation.BaSiC")
    def test_shading_correction(self, mock_basic):
        """shading_correction returns flatfield, darkfield, baseline from BaSiC fit"""
        mock_obj = MagicMock()
        mock_obj.flatfield = np.ones((5, 5))
        mock_obj.darkfield = np.zeros((5, 5))
        mock_obj.baseline = np.zeros((5,))
        mock_basic.return_value = mock_obj

        slides = [np.random.rand(5, 5) for _ in range(3)]
        shading_params = {}
        mask = np.random.rand(5, 5)

        result = shading_correction(slides, shading_params, mask)
        self.assertEqual(result["flatfield"].shape, (5, 5))
        self.assertTrue((result["darkfield"] == 0).all())
        self.assertIn("baseline", result)

    def test_unify_fields_median(self):
        """unify_fields with median mode returns float32 arrays of correct shape"""
        flatfields = [np.random.rand(5, 5) for _ in range(3)]
        darkfields = [np.random.rand(5, 5) for _ in range(3)]
        baselines = [np.random.rand(5) for _ in range(3)]

        flatfield, darkfield, baseline = unify_fields(
            flatfields, darkfields, baselines, mode="median"
        )
        self.assertEqual(flatfield.shape, (5, 5))
        self.assertEqual(darkfield.shape, (5, 5))
        self.assertEqual(baseline.shape, (5,))
        self.assertEqual(flatfield.dtype, np.float32)
        self.assertEqual(darkfield.dtype, np.float32)
        self.assertEqual(baseline.dtype, np.float32)

    def test_unify_fields_mean(self):
        """unify_fields with mean mode returns float32 arrays"""
        flatfields = [np.ones((4, 4)) * i for i in range(1, 4)]
        darkfields = [np.zeros((4, 4)) for _ in range(3)]
        baselines = [np.zeros((4,)) for _ in range(3)]

        flatfield, darkfield, baseline = unify_fields(
            flatfields, darkfields, baselines, mode="mean"
        )
        self.assertEqual(flatfield.dtype, np.float32)
        np.testing.assert_allclose(flatfield, np.full((4, 4), 2.0, dtype=np.float32))

    def test_unify_fields_invalid_mode(self):
        """unify_fields raises NotImplementedError for unknown mode"""
        flatfields = [np.random.rand(5, 5)]
        darkfields = [np.random.rand(5, 5)]
        baselines = [np.random.rand(5)]

        with self.assertRaises(NotImplementedError):
            unify_fields(flatfields, darkfields, baselines, mode="invalid")

    def test_unify_fields_mip(self):
        """unify_fields with mip mode returns max of flatfields and min of darkfields."""
        flatfields = [np.full((4, 4), float(v)) for v in [1, 3, 2]]
        darkfields = [np.full((4, 4), float(v)) for v in [1, 3, 2]]
        baselines = [np.full((4,), float(v)) for v in [1, 3, 2]]

        flatfield, darkfield, baseline = unify_fields(
            flatfields, darkfields, baselines, mode="mip"
        )
        np.testing.assert_allclose(flatfield, np.full((4, 4), 3.0, dtype=np.float32))
        np.testing.assert_allclose(darkfield, np.full((4, 4), 1.0, dtype=np.float32))
        self.assertEqual(flatfield.dtype, np.float32)

    @patch("aind_smartspim_destripe.flatfield_estimation.shading_correction")
    @patch("aind_smartspim_destripe.flatfield_estimation.filter_stripes")
    @patch("aind_smartspim_destripe.flatfield_estimation.imread")
    def test_slide_flat_estimation(self, mock_imread, mock_filter_stripes, mock_shading):
        """slide_flat_estimation returns shading results keyed by slide index."""
        img = np.ones((8, 8), dtype=np.uint16)
        mock_imread.return_value = img
        mock_filter_stripes.return_value = img
        mock_shading.return_value = {
            "flatfield": np.ones((8, 8)),
            "darkfield": np.zeros((8, 8)),
            "baseline": np.zeros((1,)),
        }

        channel_name = "Ex_488_Em_525"
        col = "X_0001"
        row_key = "X_0001_01"

        dict_struct = {channel_name: {col: {row_key: ["img_000001.tif"]}}}

        result = slide_flat_estimation(
            dict_struct=dict_struct,
            channel_name=channel_name,
            slide_idxs=[0],
            shading_parameters={},
            no_cells_config={"wavelet": "db3", "sigma": 128, "max_threshold": 12},
            cells_config={"wavelet": "db3", "sigma": 64, "max_threshold": 3},
        )

        self.assertIn(0, result)
        self.assertIn("flatfield", result[0])
        self.assertIn("darkfield", result[0])
        self.assertIn("data", result[0])
        self.assertEqual(len(result[0]["data"]), 1)

    def test_unify_fields_no_float16_underflow(self):
        """float32 cast must preserve small darkfield values that float16 would underflow"""
        small_val = 1e-5  # below float16 min positive normal (~6.1e-5)
        flatfields = [np.full((4, 4), small_val)]
        darkfields = [np.full((4, 4), small_val)]
        baselines = [np.full((4,), small_val)]

        flatfield, darkfield, baseline = unify_fields(
            flatfields, darkfields, baselines, mode="median"
        )
        self.assertTrue(
            np.all(darkfield > 0),
            "Darkfield values underflowed to zero — float32 should preserve them",
        )
