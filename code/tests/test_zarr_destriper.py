"""
Test readers
"""

import sys
import unittest

import numpy as np

sys.path.append("../")

from unittest.mock import mock_open, patch

import numpy as np

from aind_smartspim_destripe.zarr_destriper import (extract_global_to_local,
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
