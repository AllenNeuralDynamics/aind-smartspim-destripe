"""Test module for utils"""

import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, mock_open, patch

sys.path.append("../")

from aind_smartspim_destripe.utils.utils import (create_folder, get_cpu_limit,
                                                 get_memory_limit_bytes, get_size,
                                                 is_s3_path, list_s3_files,
                                                 list_s3_folders, profile_resources,
                                                 read_json_as_dict, split_s3_path,
                                                 stop_child_process)


class TestUtilities(unittest.TestCase):
    """
    Test utilities
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Setup basic job settings and job that can be used across tests"""
        cls.temp_folder = tempfile.mkdtemp(prefix="unittest_")

    @patch("os.environ.get")
    @patch("psutil.cpu_count")
    def test_get_cpu_limit(self, mock_cpu_count, mock_env_get):
        """
        Tests we get the code ocean CPU limits if
        it's a code ocean instance
        """
        mock_env_get.side_effect = lambda x: "4" if x == "CO_CPUS" else None
        mock_cpu_count.return_value = 8

        self.assertEqual(get_cpu_limit(), 4)
        self.assertIsInstance(get_cpu_limit(), int)

        mock_env_get.side_effect = lambda x: None
        with patch("builtins.open", mock_open(read_data="100000")) as mock_file:
            self.assertEqual(get_cpu_limit(), 1)

        mock_file.side_effect = FileNotFoundError
        self.assertEqual(get_cpu_limit(), 8)

    @patch("multiprocessing.Process.terminate")
    @patch("multiprocessing.Process.join")
    def test_stop_child_process(self, mock_join, mock_terminate):
        process = MagicMock()
        stop_child_process(process)
        process.terminate.assert_called_once()
        process.join.assert_called_once()

    @patch.dict(os.environ, {"AWS_BATCH_JOB_ID": "job_id"}, clear=True)
    def test_get_cpu_limit_aws_batch(self):
        """
        Tests the case where it's a pipeline execution
        """
        self.assertEqual(get_cpu_limit(), 1)

    def test_create_folder(self):
        """
        Tests the creation of a folder
        """
        with patch("os.makedirs") as mock_makedirs:
            create_folder("mock_folder", verbose=True)
            mock_makedirs.assert_called_once()

    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    @patch("os.path.exists", return_value=True)
    def test_read_json_as_dict_valid(self, mock_exists, mock_open):
        """
        Test read a valid citionari
        """
        result = read_json_as_dict("fake_path.json")
        self.assertEqual(result, {"key": "value"})

    @patch("os.path.exists", return_value=False)
    def test_read_json_as_dict_returns_empty_when_missing(self, mock_exists):
        """read_json_as_dict returns {} when the file does not exist."""
        result = read_json_as_dict("nonexistent.json")
        self.assertEqual(result, {})
        mock_exists.assert_called_once_with("nonexistent.json")

    def test_is_s3_path_true(self):
        """is_s3_path returns True for s3:// URIs."""
        self.assertTrue(is_s3_path("s3://my-bucket/folder/file.tif"))

    def test_is_s3_path_false(self):
        """is_s3_path returns False for local paths."""
        self.assertFalse(is_s3_path("/local/path/to/file.tif"))

    def test_split_s3_path(self):
        """split_s3_path separates bucket from prefix correctly."""
        bucket, prefix = split_s3_path("s3://my-bucket/folder1/folder2/")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(prefix, "folder1/folder2/")

    @patch("aind_smartspim_destripe.utils.utils.boto3")
    def test_list_s3_folders(self, mock_boto3):
        """list_s3_folders returns folder names from S3 common prefixes."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "CommonPrefixes": [
                    {"Prefix": "my/path/folder1/"},
                    {"Prefix": "my/path/folder2/"},
                ]
            }
        ]

        result = list_s3_folders("my-bucket", "my/path/")
        self.assertEqual(sorted(result), ["folder1", "folder2"])

    @patch("aind_smartspim_destripe.utils.utils.boto3")
    def test_list_s3_files(self, mock_boto3):
        """list_s3_files returns keys matching the given extension."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "my/path/image.tif"},
                    {"Key": "my/path/image.png"},
                ]
            }
        ]

        result = list_s3_files("my-bucket", "my/path/", extension=".tif")
        self.assertEqual(result, ["my/path/image.tif"])

    def test_get_size_bytes(self):
        """get_size formats byte counts into human-readable strings."""
        self.assertEqual(get_size(0), "0.00B")
        self.assertEqual(get_size(1024), "1.00KB")
        self.assertEqual(get_size(1024**2), "1.00MB")

    @patch.dict(os.environ, {"CO_MEMORY": "16"}, clear=False)
    def test_get_memory_limit_bytes_co_env(self):
        """get_memory_limit_bytes reads CO_MEMORY when set."""
        result = get_memory_limit_bytes()
        self.assertEqual(result, 16)

    @patch("psutil.virtual_memory")
    def test_get_memory_limit_bytes_psutil_fallback(self, mock_vmem):
        """get_memory_limit_bytes falls back to psutil when no env vars are set."""
        mock_vmem.return_value = MagicMock(total=8 * 1024**3)
        with patch.dict(os.environ, {}, clear=True):
            result = get_memory_limit_bytes()
        self.assertEqual(result, 8 * 1024**3)

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down class method to clean up"""
        if os.path.exists(cls.temp_folder):
            shutil.rmtree(cls.temp_folder, ignore_errors=True)
