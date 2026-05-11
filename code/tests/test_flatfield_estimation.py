"""Tests for flatfield estimation"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.append("../")
from aind_smartspim_destripe.flatfield_estimation import shading_correction, unify_fields


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
