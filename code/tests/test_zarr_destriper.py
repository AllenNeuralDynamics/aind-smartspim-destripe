"""
Test readers
"""

import sys
import unittest

import numpy as np

sys.path.append("../")

from unittest.mock import MagicMock, mock_open, patch

import numpy as np

from aind_smartspim_destripe.zarr_destriper import (extract_global_to_local,
                                                    get_microscope_flats,
                                                    pad_array_n_d,
                                                    read_json_as_dict)


class TestZarrDestriper(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    @patch("os.path.exists", return_value=True)
    def test_read_json_as_dict_valid(self, mock_exists, mock_open):
        """
        Test read a valid citionari
        """
        result = read_json_as_dict("fake_path.json")
        self.assertEqual(result, {"key": "value"})

    @patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "error"))
    @patch("os.path.exists", return_value=True)
    def test_read_json_as_dict_unicode_error(self, mock_exists, mock_open):
        """
        Test read json when unicode error
        """
        with self.assertRaises(UnicodeDecodeError):
            read_json_as_dict("fake_path.json")

    @patch("os.path.exists", return_value=False)
    def test_read_json_as_dict_file_not_found(self, mock_exists):
        """
        Reads json when it does not exist
        """
        result = read_json_as_dict("fake_path.json")
        self.assertEqual(result, {})

    def test_pad_array_n_d(self):
        arr = np.zeros((3, 3))
        padded = pad_array_n_d(arr, dim=5)
        self.assertEqual(padded.shape, (1, 1, 1, 3, 3))

    def test_extract_global_to_local(self):
        """
        Test extract global ids to local
        """
        global_ids = np.array([[10, 20, 30, 1], [40, 50, 60, 2]])
        global_slices = (slice(5, 15), slice(15, 25), slice(25, 35))
        local_ids = extract_global_to_local(global_ids, global_slices)
        self.assertTrue(np.all(local_ids[:, :3] >= 0))

    def test_extract_global_to_local_with_pad(self):
        """extract_global_to_local maps coordinates correctly when pad > 0."""
        global_ids = np.array([[10, 20, 30, 1]])
        global_slices = (slice(5, 15), slice(15, 25), slice(25, 35))
        local_ids = extract_global_to_local(global_ids, global_slices, pad=2)
        # start_pos = [3, 13, 23], local = [10-3-2, 20-13-2, 30-23-2] = [5, 5, 5]
        self.assertEqual(len(local_ids), 1)
        np.testing.assert_array_equal(local_ids[0, :3], [5, 5, 5])

    def test_pad_array_n_d_already_5d(self):
        """pad_array_n_d leaves a 5D array unchanged."""
        arr = np.zeros((1, 1, 2, 3, 4))
        padded = pad_array_n_d(arr, dim=5)
        self.assertEqual(padded.shape, (1, 1, 2, 3, 4))

    def test_pad_array_n_d_dim_exceeds_max(self):
        """pad_array_n_d raises ValueError when dim > 5."""
        arr = np.zeros((3, 3))
        with self.assertRaises(ValueError):
            pad_array_n_d(arr, dim=6)

    def test_get_microscope_flats_no_metadata(self):
        """get_microscope_flats returns (None, None) when metadata.json is absent."""
        derivatives_folder = MagicMock()
        derivatives_folder.joinpath.return_value.exists.return_value = False

        flatfield, metadata = get_microscope_flats("Ex_488_Em_525", derivatives_folder)
        self.assertIsNone(flatfield)
        self.assertIsNone(metadata)

    @patch("aind_smartspim_destripe.zarr_destriper.os.path.exists", return_value=True)
    @patch("aind_smartspim_destripe.zarr_destriper.tif")
    @patch("aind_smartspim_destripe.zarr_destriper.glob")
    @patch("aind_smartspim_destripe.zarr_destriper.utils.read_json_as_dict")
    def test_get_microscope_flats_valid(
        self, mock_read_json, mock_glob, mock_tif, mock_os_exists
    ):
        """get_microscope_flats parses tile_config and loads two hemisphere flatfields."""
        mock_read_json.return_value = {
            "tile_config": {
                "0": {"Laser": "488", "X": "X_0001", "Y": "X_0001_Y_0001", "Side": 0},
                "1": {"Laser": "488", "X": "X_0002", "Y": "X_0002_Y_0001", "Side": 1},
            }
        }
        mock_glob.return_value = [
            "/flat/FlatReal488_left.tif",
            "/flat/FlatReal488_right.tif",
        ]
        mock_tif.imread.return_value = np.ones((50, 50), dtype=np.uint16)

        derivatives_folder = MagicMock()
        derivatives_folder.joinpath.return_value.exists.return_value = True

        flatfields, metadata = get_microscope_flats("Ex_488_Em_525", derivatives_folder)

        self.assertEqual(len(flatfields), 2)
        self.assertIn("X_0001", metadata)
        self.assertIn("X_0002", metadata)
        mock_os_exists.assert_called()
