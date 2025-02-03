"""
Tests flatfield estimation, commented out
until fixing imports
"""

# import unittest
# from unittest.mock import patch, MagicMock
# import numpy as np
# import sys
# sys.path.append("../")
# from aind_smartspim_destripe.flatfield_estimation import (
#     shading_correction,
#     flatfield_correction,
#     create_median_flatfield,
#     estimate_flats_per_laser,
#     unify_fields,
# )

# class TestShadingCorrectionFunctions(unittest.TestCase):
#     @patch("aind_smartspim_flatfield_estimation.flatfield_estimation.BaSiC")
#     def test_shading_correction(self, mock_basic):
#         # Mock BaSiC behavior
#         mock_obj = MagicMock()
#         mock_obj.flatfield = np.ones((5, 5))
#         mock_obj.darkfield = np.zeros((5, 5))
#         mock_obj.baseline = np.zeros((5,))
#         mock_basic.return_value = mock_obj

#         slides = [np.random.rand(5, 5) for _ in range(3)]
#         shading_params = {"param1": 1, "param2": 2}
#         mask = np.random.rand(5, 5)

#         result = shading_correction(slides, shading_params, mask)
#         self.assertEqual(result["flatfield"].shape, (5, 5))
#         self.assertTrue((result["darkfield"] == 0).all())

#     def test_flatfield_correction(self):
#         tiles = [np.random.randint(100, 255, (5, 5), dtype="uint16") for _ in range(3)]
#         flatfield = np.ones((5, 5), dtype="float32") * 2
#         darkfield = np.zeros((5, 5), dtype="float32")
#         baseline = np.zeros((3,))

#         result = flatfield_correction(tiles, flatfield, darkfield, baseline)
#         self.assertEqual(result.shape, (3, 5, 5))
#         self.assertTrue(result.dtype == np.uint16)

#     def test_create_median_flatfield(self):
#         flatfield = np.random.rand(10, 10)
#         result = create_median_flatfield(flatfield, smooth=True)
#         self.assertEqual(result.shape, (10, 10))
#         self.assertTrue(np.isclose(result.mean(), flatfield.mean(), atol=0.5))

#     @patch("aind_smartspim_flatfield_estimation.flatfield_estimation.shading_correction")
#     def test_estimate_flats_per_laser(self, mock_shading_correction):
#         mock_shading_correction.return_value = {
#             "flatfield": np.ones((5, 5)),
#             "darkfield": np.zeros((5, 5)),
#             "baseline": np.zeros((5,)),
#         }

#         tiles_per_side = {
#             "left": [np.random.rand(5, 5) for _ in range(3)],
#             "right": [np.random.rand(5, 5) for _ in range(3)],
#         }
#         shading_params = {"param1": 1}

#         result = estimate_flats_per_laser(tiles_per_side, shading_params)
#         self.assertIn("left", result)
#         self.assertIn("right", result)
#         self.assertEqual(result["left"]["flatfield"].shape, (5, 5))

#     def test_unify_fields(self):
#         flatfields = [np.random.rand(5, 5) for _ in range(3)]
#         darkfields = [np.random.rand(5, 5) for _ in range(3)]
#         baselines = [np.random.rand(5,) for _ in range(3)]

#         flatfield, darkfield, baseline = unify_fields(
#             flatfields, darkfields, baselines, mode="median"
#         )
#         self.assertEqual(flatfield.shape, (5, 5))
#         self.assertEqual(darkfield.shape, (5, 5))
#         self.assertEqual(baseline.shape, (5,))
#         self.assertTrue(flatfield.dtype == np.float16)

#         with self.assertRaises(NotImplementedError):
#             unify_fields(flatfields, darkfields, baselines, mode="invalid")
