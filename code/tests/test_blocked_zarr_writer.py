"""Tests for the smartspim filtering"""

import sys
import unittest

sys.path.append("../")
from aind_smartspim_destripe import blocked_zarr_writer


class TestBlockedArrayWriter(unittest.TestCase):

    def test_get_size(self):
        """
        Test getting dataset size
        """
        self.assertEqual(blocked_zarr_writer._get_size((2, 3), 4), 24)
        self.assertEqual(blocked_zarr_writer._get_size((1, 1, 1), 2), 2)
        with self.assertRaises(ValueError):
            blocked_zarr_writer._get_size((2, -3), 4)

    def test_closer_to_target(self):
        """
        Tests getting the shape closer to target
        based on target size
        """
        shape1 = (4, 4)
        shape2 = (8, 8)
        itemsize = 1
        target_bytes = 30
        self.assertEqual(
            blocked_zarr_writer._closer_to_target(
                shape1, shape2, target_bytes, itemsize
            ),
            shape1,
        )
        target_bytes = 60
        self.assertEqual(
            blocked_zarr_writer._closer_to_target(
                shape1, shape2, target_bytes, itemsize
            ),
            shape2,
        )

    def test_expand_chunks_cycle(self):
        """
        Tests expanding chunks based on modes
        """
        chunks = (2, 2, 2)
        data_shape = (16, 16, 16)
        target_size = 64
        itemsize = 1
        result = blocked_zarr_writer.expand_chunks(
            chunks, data_shape, target_size, itemsize, mode="cycle"
        )
        self.assertEqual(result, (4, 4, 4))
        with self.assertRaises(ValueError):
            blocked_zarr_writer.expand_chunks(
                (0, 2, 2), data_shape, target_size, itemsize
            )

    def test_expand_chunks_iso(self):
        """
        Tests expanding chunks based on modes
        """
        chunks = (2, 2, 2)
        data_shape = (16, 16, 16)
        target_size = 64
        itemsize = 1
        result = blocked_zarr_writer.expand_chunks(
            chunks, data_shape, target_size, itemsize, mode="iso"
        )
        self.assertEqual(result, (4, 4, 4))
        with self.assertRaises(ValueError):
            blocked_zarr_writer.expand_chunks(
                (0, 2, 2), data_shape, target_size, itemsize
            )

    def test_gen_slices(self):
        """
        Tests getting dataset slices
        """
        arr_shape = (5, 5)
        block_shape = (2, 2)
        slices = list(
            blocked_zarr_writer.BlockedArrayWriter.gen_slices(arr_shape, block_shape)
        )
        self.assertEqual(len(slices), 9)  # 3x3 blocks
        self.assertEqual(slices[0], (slice(0, 2), slice(0, 2)))
        self.assertEqual(slices[-1], (slice(4, 5), slice(4, 5)))
