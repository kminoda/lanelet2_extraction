import unittest
from lanelet2_extraction.extractor import filter_polyline_by_roi_size
from numpy.testing import assert_array_equal
import numpy as np

class TestLanelet2Extractor(unittest.TestCase):
    def test_filter_polyline_by_roi_size_entirely_within_roi(self):
        polyline = [(10, 10), (20, 20), (30, 30)]
        roi_size = (80, 80)
        expected_result = [
            [[10, 10], [20, 20], [30, 30]],
        ]

        result = filter_polyline_by_roi_size(polyline, roi_size)
        self.assertEqual(len(result), len(expected_result))
        for i in range(len(result)):
            assert_array_equal(np.array(result[i]), np.array(expected_result[i]))

    def test_filter_polyline_by_roi_size_partially_within_roi_1(self):
        polyline = [(10, 10), (50, 50), (90, 90)]
        roi_size = (120, 120)
        expected_result = [
            [[10, 10], [50, 50], [60, 60]],
        ]

        result = filter_polyline_by_roi_size(polyline, roi_size)
        self.assertEqual(len(result), len(expected_result))
        for i in range(len(result)):
            assert_array_equal(np.array(result[i]), np.array(expected_result[i]))

    def test_filter_polyline_by_roi_size_partially_within_roi_2(self):
        polyline = [(10, 10), (90, 90)]
        roi_size = (120, 120)
        expected_result = [
            [[10, 10], [60, 60]],
        ]

        result = filter_polyline_by_roi_size(polyline, roi_size)
        self.assertEqual(len(result), len(expected_result))
        for i in range(len(result)):
            assert_array_equal(np.array(result[i]), np.array(expected_result[i]))

    def test_filter_polyline_by_roi_size_partially_within_roi_3(self):
        polyline = [(-90, 0), (90, 0)]
        roi_size = (120, 120)
        expected_result = [
            [[-60, 0], [60, 0]],
        ]

        result = filter_polyline_by_roi_size(polyline, roi_size)
        self.assertEqual(len(result), len(expected_result))
        for i in range(len(result)):
            assert_array_equal(np.array(result[i]), np.array(expected_result[i]))

    def test_filter_polyline_by_roi_size_partially_within_roi_4(self):
        polyline = [(10, 90), (10, -90), (-10, -90), (-10, 90)]
        roi_size = (120, 120)
        expected_result = [
            [[10, 60], [10, -60]],
            [[-10, -60], [-10, 60]]
        ]

        result = filter_polyline_by_roi_size(polyline, roi_size)
        self.assertEqual(len(result), len(expected_result))
        for i in range(len(result)):
            assert_array_equal(np.array(result[i]), np.array(expected_result[i]))

    def test_filter_polyline_by_roi_size_entirely_outside_roi_1(self):
        polyline = [(80, 80), (90, 90), (100, 100)]
        roi_size = (120, 120)
        expected_result = []

        result = filter_polyline_by_roi_size(polyline, roi_size)
        self.assertEqual(len(result), len(expected_result))
        for i in range(len(result)):
            assert_array_equal(np.array(result[i]), np.array(expected_result[i]))

    def test_filter_polyline_by_roi_size_entirely_outside_roi_2(self):
        polyline = [(0, 200), (200, 0)]
        roi_size = (120, 120)
        expected_result = []

        result = filter_polyline_by_roi_size(polyline, roi_size)
        self.assertEqual(len(result), len(expected_result))
        for i in range(len(result)):
            assert_array_equal(np.array(result[i]), np.array(expected_result[i]))

if __name__ == "__main__":
    unittest.main()