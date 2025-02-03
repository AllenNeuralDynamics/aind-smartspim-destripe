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

from aind_smartspim_destripe.utils.utils import (create_folder,
                                                 get_code_ocean_cpu_limit,
                                                 profile_resources,
                                                 read_json_as_dict,
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
    def test_get_code_ocean_cpu_limit(self, mock_cpu_count, mock_env_get):
        """
        Tests we get the code ocean CPU limits if
        it's a code ocean instance
        """
        mock_env_get.side_effect = lambda x: "4" if x == "CO_CPUS" else None
        mock_cpu_count.return_value = 8

        self.assertEqual(get_code_ocean_cpu_limit(), "4")

        mock_env_get.side_effect = lambda x: None
        with patch("builtins.open", mock_open(read_data="100000")) as mock_file:
            self.assertEqual(get_code_ocean_cpu_limit(), 1)

        mock_file.side_effect = FileNotFoundError
        self.assertEqual(get_code_ocean_cpu_limit(), 8)

    @patch("multiprocessing.Process.terminate")
    @patch("multiprocessing.Process.join")
    def test_stop_child_process(self, mock_join, mock_terminate):
        process = MagicMock()
        stop_child_process(process)
        process.terminate.assert_called_once()
        process.join.assert_called_once()

    @patch.dict(os.environ, {"AWS_BATCH_JOB_ID": "job_id"}, clear=True)
    def test_get_code_ocean_cpu_limit_aws_batch(self):
        """
        Tests the case where it's a pipeline execution
        """
        self.assertEqual(get_code_ocean_cpu_limit(), 1)

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

    @patch("psutil.cpu_percent", side_effect=[10, 20, 30])
    @patch(
        "psutil.virtual_memory",
        side_effect=[
            MagicMock(percent=50),
            MagicMock(percent=60),
            MagicMock(percent=70),
        ],
    )
    def test_profile_resources(self, mock_virtual_memory, mock_cpu_percent):
        time_points = []
        cpu_percentages = []
        memory_usages = []
        monitoring_interval = 1

        def run_profiler():
            profile_resources(
                time_points, cpu_percentages, memory_usages, monitoring_interval
            )

        profiler_thread = threading.Thread(target=run_profiler, daemon=True)
        profiler_thread.start()

        time.sleep(3.5)

        self.assertGreater(len(time_points), 0)
        self.assertGreater(len(cpu_percentages), 0)
        self.assertGreater(len(memory_usages), 0)

        self.assertEqual(cpu_percentages[:3], [10, 20, 30])
        self.assertEqual(memory_usages[:3], [50, 60, 70])

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down class method to clean up"""
        if os.path.exists(cls.temp_folder):
            shutil.rmtree(cls.temp_folder, ignore_errors=True)
