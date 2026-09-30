import unittest

import numpy as np

from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
)
from numpy_tasks import (
    analyze_time_series,
    binarize,
    chess,
    draw_ellipse,
    draw_rectangle,
    matrix_statistics,
    one_hot,
    sum_prod,
    unique_columns,
    unique_rows,
)


class NumpyTasksTests(unittest.TestCase):
    def test_sum_prod(self):
        matrices = np.array([
            [[1, 0], [0, 1]],
            [[2, 0], [0, 2]],
        ])
        vectors = np.array([
            [[1], [2]],
            [[3], [4]],
        ])
        result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
        np.testing.assert_array_equal(result, np.array([[7], [10]]))

    def test_binarize(self):
        matrix = np.array([[0.2, 0.6], [0.5, 1.0]])
        result = binarize(BinarizeInput(matrix, 0.5))
        np.testing.assert_array_equal(result, np.array([[0, 1], [0, 1]]))
        np.testing.assert_array_equal(matrix, np.array([[0.2, 0.6], [0.5, 1.0]]))

    def test_unique_rows(self):
        matrix = np.array([[2, 1, 2], [3, 3, 1]])
        self.assertEqual(unique_rows(MatrixInput(matrix)), [[1, 2], [1, 3]])

    def test_unique_columns(self):
        matrix = np.array([[1, 2], [1, 3], [2, 3]])
        self.assertEqual(unique_columns(MatrixInput(matrix)), [[1, 2], [2, 3]])

    def test_matrix_statistics(self):
        stats = matrix_statistics(RandomMatrixInput(3, 4, mean=2.0, std=0.5, seed=10))
        self.assertEqual(stats.matrix.shape, (3, 4))
        np.testing.assert_allclose(stats.row_means, np.mean(stats.matrix, axis=1))
        np.testing.assert_allclose(stats.column_means, np.mean(stats.matrix, axis=0))
        np.testing.assert_allclose(stats.row_variances, np.var(stats.matrix, axis=1))
        np.testing.assert_allclose(stats.column_variances, np.var(stats.matrix, axis=0))

    def test_chess(self):
        result = chess(ChessInput(3, 4, 0, 1))
        expected = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_draw_rectangle(self):
        result = draw_rectangle(
            RectangleInput(
                width=2,
                height=2,
                image_height=4,
                image_width=6,
                shape_color=(255, 0, 0),
                background_color=(0, 0, 0),
            )
        )
        self.assertEqual(result.shape, (4, 6, 3))
        np.testing.assert_array_equal(result[1:3, 2:4], np.full((2, 2, 3), (255, 0, 0), dtype=np.uint8))

    def test_draw_ellipse(self):
        result = draw_ellipse(
            EllipseInput(
                semi_axis_x=2,
                semi_axis_y=1,
                image_height=5,
                image_width=5,
                shape_color=(255, 255, 255),
                background_color=(0, 0, 0),
            )
        )
        self.assertEqual(result.shape, (5, 5, 3))
        np.testing.assert_array_equal(result[2, 2], np.array([255, 255, 255]))

    def test_time_series(self):
        stats = analyze_time_series(TimeSeriesInput([1, 3, 2, 4, 1], 3))
        self.assertAlmostEqual(stats.mean, 2.2)
        self.assertAlmostEqual(stats.variance, np.var([1, 3, 2, 4, 1]))
        self.assertAlmostEqual(stats.std, np.std([1, 3, 2, 4, 1]))
        np.testing.assert_array_equal(stats.local_maxima_indices, np.array([1, 3]))
        np.testing.assert_array_equal(stats.local_minima_indices, np.array([2]))
        np.testing.assert_allclose(stats.moving_average, np.array([2.0, 3.0, 7 / 3]))

    def test_one_hot(self):
        result = one_hot(OneHotInput([0, 2, 3, 0]))
        expected = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ])
        np.testing.assert_array_equal(result, expected)

        result_with_count = one_hot(OneHotInput([0, 2], class_count=5))
        self.assertEqual(result_with_count.shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
