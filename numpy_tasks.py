import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices = np.asarray(data.matrices)
    vectors = np.asarray(data.vectors)

    if vectors.ndim == 2:
        vectors = vectors[..., np.newaxis]

    products = np.matmul(matrices, vectors)
    return np.sum(products, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix = np.asarray(data.matrix)
    return (matrix > data.threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(column).tolist() for column in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed

    rng = np.random.default_rng(seed)
    matrix = rng.normal(mean, std, size=(rows, columns))

    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )


def plot_matrix_histograms(matrix: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix)

    for row_index, row in enumerate(matrix):
        plt.figure()
        plt.hist(row)
        plt.title(f"Row {row_index}")

    for column_index, column in enumerate(matrix.T):
        plt.figure()
        plt.hist(column)
        plt.title(f"Column {column_index}")

    plt.show()


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    indices = np.indices((rows, columns))
    return np.where((indices[0] + indices[1]) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color

    start_x = (image_width - width) // 2
    start_y = (image_height - height) // 2
    image[start_y:start_y + height, start_x:start_x + width] = shape_color

    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color

    center_x = image_width // 2
    center_y = image_height // 2

    y, x = np.ogrid[:image_height, :image_width]
    mask = (
        ((x - center_x) ** 2) / (semi_axis_x ** 2)
        + ((y - center_y) ** 2) / (semi_axis_y ** 2)
        <= 1
    )
    image[mask] = shape_color

    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values = np.asarray(data.values, dtype=float)
    window = data.window

    if window <= 0 or window > len(values):
        raise ValueError("Window must be between 1 and the series length")

    local_maxima = np.where(
        (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
    )[0] + 1

    local_minima = np.where(
        (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
    )[0] + 1

    moving_average = np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )

    return TimeSeriesStatistics(
        mean=float(np.mean(values)),
        variance=float(np.var(values)),
        std=float(np.std(values)),
        local_maxima_indices=local_maxima,
        local_minima_indices=local_minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.asarray(data.labels, dtype=int)

    if labels.ndim != 1:
        raise ValueError("Labels must be one-dimensional")
    if np.any(labels < 0):
        raise ValueError("Labels must be non-negative")

    if data.class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0
    else:
        class_count = data.class_count

    if class_count < 0:
        raise ValueError("Class count must be non-negative")
    if labels.size and labels.max() >= class_count:
        raise ValueError("Class count is too small for the labels")

    result = np.zeros((labels.size, class_count), dtype=int)
    if labels.size:
        result[np.arange(labels.size), labels] = 1

    return result
