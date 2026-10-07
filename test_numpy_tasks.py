import unittest
from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt

from numpy_tasks import (
    sum_prod,
    binarize,
    unique_rows,
    unique_columns,
    matrix_statistics,
    plot_matrix_histograms,
    chess,
    draw_rectangle,
    draw_ellipse,
    analyze_time_series,
    one_hot,
)

from grader_contracts.numpy_tasks import (
    MatrixVectorBatchInput,
    BinarizeInput,
    MatrixInput,
    RandomMatrixInput,
    ChessInput,
    RectangleInput,
    EllipseInput,
    TimeSeriesInput,
    OneHotInput,
)


class TestNumpyTasks(unittest.TestCase):

    def test_sum_prod(self):
        matrices = [
            np.array([[1, 0], [0, 1]]),
            np.array([[2, 0], [0, 2]])]
        vectors = [
            np.array([[1], [2]]),
            np.array([[3], [4]])]
        result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
        expected = np.array([
            [7],
            [10]
        ])
        np.testing.assert_array_equal(result, expected)

    def test_binarize(self):
        matrix = np.array([
            [0.2, 0.8],
            [0.5, 1.0]])

        result = binarize(BinarizeInput(matrix, 0.5))
        expected = np.array([
            [0, 1],
            [0, 1]
        ])
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(matrix,np.array([[0.2, 0.8], [0.5, 1.0]]))

    def test_unique_rows(self):
        matrix = np.array([
            [1, 2, 1],
            [3, 2, 3],
            [1, 4, 4]
        ])

        result = unique_rows(MatrixInput(matrix))
        self.assertEqual(result, [[1, 2], [2, 3], [1, 4]])

    def test_unique_columns(self):
        matrix = np.array([
            [1, 2, 1],
            [3, 2, 3],
            [1, 4, 4]])
        result = unique_columns(MatrixInput(matrix))
        self.assertEqual(result,[[1, 3], [2, 4], [1, 3, 4]])

    def test_matrix_statistics(self):
        result = matrix_statistics(RandomMatrixInput(3, 4, 0, 1, 42))
        self.assertEqual(result.matrix.shape, (3, 4))
        np.testing.assert_allclose(
            result.row_means,
            np.mean(result.matrix, axis=1))
        np.testing.assert_allclose(
            result.column_means,
            np.mean(result.matrix, axis=0))
        np.testing.assert_allclose(
            result.row_variances,
            np.var(result.matrix, axis=1))
        np.testing.assert_allclose(
            result.column_variances,
            np.var(result.matrix, axis=0))

    def test_matrix_statistics_same_seed(self):
        first = matrix_statistics(
            RandomMatrixInput(2, 3, 0, 1, 42))

        second = matrix_statistics(
            RandomMatrixInput(2, 3, 0, 1, 42))

        np.testing.assert_array_equal(
            first.matrix,
            second.matrix)
    @patch("numpy_tasks.plt.show")

    def test_plot_matrix_histograms(self, mock_show):
        matrix = np.array([
            [1, 2, 3],
            [4, 5, 6]])

        plot_matrix_histograms(matrix)
        self.assertEqual(mock_show.call_count, 5)
        plt.close("all")

    def test_chess(self):
        result = chess(
            ChessInput(3, 4, 0, 1))
        expected = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_draw_rectangle(self):
        result = draw_rectangle(
            RectangleInput(
                4,
                2,
                6,
                8,
                (255, 0, 0),
                (255, 255, 255)
            )
        )
        self.assertEqual(result.shape, (6, 8, 3))
        red_pixels = np.sum(
            np.all(result == [255, 0, 0], axis=2))
        self.assertEqual(red_pixels, 8)

    def test_draw_ellipse(self):
        result = draw_ellipse(
            EllipseInput(
                3,
                2,
                7,
                9,
                (255, 0, 0),
                (255, 255, 255)
            )
        )

        self.assertEqual(result.shape, (7, 9, 3))
        red_pixels = np.sum(np.all(result == [255, 0, 0], axis=2))
        self.assertGreater(red_pixels, 0)

    def test_analyze_time_series(self):
        values = np.array([1, 3, 2, 5, 4])
        result = analyze_time_series(
            TimeSeriesInput(values, 3))
        self.assertAlmostEqual(result.mean, 3.0)
        self.assertAlmostEqual(result.variance, 2.0)
        self.assertAlmostEqual(result.std, np.sqrt(2))
        self.assertEqual(
            result.local_maxima_indices,
            [1, 3])
        self.assertEqual(
            result.local_minima_indices,
            [2])
        np.testing.assert_allclose(
            result.moving_average,
            [2.0, 10 / 3, 11 / 3])

    def test_one_hot(self):
        labels = np.array([0, 2, 3, 0])
        result = one_hot(
            OneHotInput(labels))

        expected = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_one_hot_with_class_count(self):
        result = one_hot(
            OneHotInput(
                np.array([0, 2]),
                5
            )
        )

        expected = np.array([
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ])
        np.testing.assert_array_equal(result, expected)
if __name__ == "__main__":
    unittest.main()