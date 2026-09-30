import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices = np.asarray(data.matrices)
    vectors = np.asarray(data.vectors)
    products = np.matmul(matrices, vectors)
    return np.sum(products, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix = np.asarray(data.matrix)
    return (matrix > data.threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    result = []

    for row in matrix:
        result.append(np.unique(row).tolist())

    return result


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    result = []

    for column in matrix.T:
        result.append(np.unique(column).tolist())

    return result


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rng = np.random.default_rng(data.seed)
    matrix = rng.normal(data.mean, data.std, (data.rows, data.columns))

    row_means = np.mean(matrix, axis=1)
    column_means = np.mean(matrix, axis=0)
    row_variances = np.var(matrix, axis=1)
    column_variances = np.var(matrix, axis=0)

    return MatrixStatistics(
        matrix,
        row_means,
        column_means,
        row_variances,
        column_variances,
    )


def plot_matrix_histograms(matrix: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    for row in matrix:
        plt.figure()
        plt.hist(row)

    for column in matrix.T:
        plt.figure()
        plt.hist(column)

    plt.show()


def chess(data: ChessInput) -> np.ndarray:
    result = np.empty((data.rows, data.columns))

    for i in range(data.rows):
        for j in range(data.columns):
            if (i + j) % 2 == 0:
                result[i, j] = data.first
            else:
                result[i, j] = data.second

    return result


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    image = np.empty((data.image_height, data.image_width, 3), dtype=np.uint8)
    image[:] = data.background_color

    x = (data.image_width - data.width) // 2
    y = (data.image_height - data.height) // 2

    image[y:y + data.height, x:x + data.width] = data.shape_color

    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    image = np.empty((data.image_height, data.image_width, 3), dtype=np.uint8)
    image[:] = data.background_color

    center_x = data.image_width // 2
    center_y = data.image_height // 2

    y, x = np.ogrid[:data.image_height, :data.image_width]

    mask = (
        ((x - center_x) ** 2) / data.semi_axis_x ** 2
        + ((y - center_y) ** 2) / data.semi_axis_y ** 2
        <= 1
    )

    image[mask] = data.shape_color

    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values = np.asarray(data.values, dtype=float)

    maximums = []
    minimums = []

    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            maximums.append(i)

        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            minimums.append(i)

    moving_average = []

    for i in range(len(values) - data.window + 1):
        moving_average.append(np.mean(values[i:i + data.window]))

    return TimeSeriesStatistics(
        float(np.mean(values)),
        float(np.var(values)),
        float(np.std(values)),
        np.array(maximums),
        np.array(minimums),
        np.array(moving_average),
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.asarray(data.labels, dtype=int)

    if data.class_count is None:
        class_count = int(np.max(labels)) + 1
    else:
        class_count = data.class_count

    result = np.zeros((len(labels), class_count), dtype=int)

    for i in range(len(labels)):
        result[i, labels[i]] = 1

    return result
