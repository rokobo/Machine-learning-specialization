"""Helper functions for Jupyter notebooks."""
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_data(
    x: ndarray, y: ndarray, title: str = "",
    y_title: str = "Y values", x_title: str = "X values",
    colors: ndarray | str = "r", labels: dict[any, str] = None,
    margin_scale: float = 0.025
) -> None:
    """
    Generates scatter plot for data visualization

    Args:
        x (ndarray): x values.
        y (ndarray): y values.
        title (str, optional): Graph title. Defaults to "".
        y_title (str, optional): Y values label. Defaults to "Y values".
        x_title (str, optional): X values label. Defaults to "X values".
        colors (ndarray | str, optional): Val to assign color. Defaults to "r".
        labels (dict[any, str], optional): Color val: label. Defaults to None.
        margin_scale (float, optional): Margin of the graph. Defaults to 0.025
    """
    # Check if labels are being used
    if isinstance(colors, ndarray) and labels is not None:
        # Plot each scatter plot based on color value
        unique_colors = np.unique(colors)
        for color in unique_colors:
            mask = colors == color
            filtered_x = x[mask]
            filtered_y = y[mask]
            plt.scatter(filtered_x, filtered_y, label=labels[color])
            plt.legend()
    else:
        # Plot simple scatter
        plt.scatter(x, y, marker='x', c=colors)

    plt.title(title)
    plt.ylabel(y_title)
    plt.xlabel(x_title)

    # Trim graph
    max_x, min_x, max_y, min_y = max(x), min(x), max(y), min(y)
    x_margin, y_margin = max_x * margin_scale, max_y * margin_scale
    plt.xlim(min_x - x_margin, max_x + x_margin)
    plt.ylim(min_y - y_margin, max_y + y_margin)


def plot_prediction_line(x: ndarray, y: ndarray, color: str = "b"):
    """
    Plots prediction line.

    Args:
        x (ndarray): X values.
        y (ndarray): Y values.
        color (str, optional): Color of the line. Defaults to "b".
    """
    plt.plot(x, y, color=color)


def sigmoid(z: ndarray) -> ndarray:
    """
    Computes the sigmoid of all values of the z array.

    Args:
        z (ndarray): Z array.

    Returns:
        ndarray: Sigmoid function applied to all values of z.
    """
    return 1 / (1 + np.exp(-z))


def feature_map(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Applies feature mapping up to sixth degree.

    Args:
        x1 (ndarray): x1 dataset.
        x2 (ndarray): x2 dataset.

    Returns:
        ndarray: New dataset with feature mapping.
    """
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    out = []
    for i in range(1, 7):
        for j in range(i + 1):
            out.append((x1**(i-j) * (x2**j)))
    return np.stack(out, axis=1)


def plot_boundary_line(
        x: ndarray, w: ndarray, b: float, color: str = "r"):
    """
    Plots boundary line.

    Args:
        x (ndarray): X values.
        w (ndarray): W parameters.
        b (float): B parameter.
        color (str, optional): Color of the line. Defaults to "r".
    """
    if x.shape[1] <= 2:
        plot_x = np.array([min(x[:, 0]), max(x[:, 0])])
        plot_y = (-1.0 / w[1]) * (w[0] * plot_x + b)
        plt.plot(plot_x, plot_y, '--', color=color)
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))

        for i, u_val in enumerate(u):
            for j, v_val in enumerate(v):
                z[i, j] = sigmoid(np.dot(feature_map(u_val, v_val), w) + b)

        z = z.T
        plt.contour(u, v, z, levels=[0.5], linestyles='--', colors=color)


def plot_drawings(
    x: ndarray, y: ndarray, size: int = 8,
    model=None, model2=None, model3=None
):
    """
    Plots drawings.

    Args:
        x (ndarray): x values.
        y (ndarray): y values.
        size (int): How many pictures to show.
        model (optional): Prediction model. Defaults to None.
        model2 (optional): Prediction model2. Defaults to None.
        model3 (optional): Prediction model3. Defaults to None.
    """
    m, _ = x.shape

    fig, axes = plt.subplots(size, size, figsize=(size, size))
    fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92])

    for _, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        x_random_reshaped = x[random_index].reshape((20, 20)).T

        # Display the image
        ax.imshow(x_random_reshaped, cmap='gray')

        # Display the label above the image
        label = str(y[random_index, 0])
        if model is not None:
            prediction = model.predict(
                x[random_index].reshape(1, 400), verbose=None
            )
            yhat = (1 if prediction >= 0.5 else 0)
            label += f", {yhat}"
        if model2 is not None:
            prediction = model2.predict(x[random_index])
            yhat = (1 if prediction >= 0.5 else 0)
            label += f", {yhat}"
        if model3 is not None:
            prediction = model3.predict(
                x[random_index].reshape(1, 400), verbose=None
            )
            prediction_p = tf.nn.softmax(prediction)
            yhat = np.argmax(prediction_p)
            label += f", {yhat}"
        ax.set_title(label)
        ax.set_axis_off()

    # Add title
    title = "Label"
    num = 0
    title += ", prediction " + str(num := num+1) if model is not None else ""
    title += ", prediction " + str(num := num+1) if model2 is not None else ""
    title += ", prediction " + str(num := num+1) if model3 is not None else ""
    fig.suptitle(title, fontsize=16)


def get_error_rate(model, x: ndarray, y: ndarray) -> None:
    """
    Prints error rate for softmax model.

    Args:
        model: Softmax model
        x (ndarray): x values.
        y (ndarray): y values.

    Returns:
        int: _description_
    """
    predictions = model.predict(x)
    yhat = np.argmax(predictions, axis=1)
    errors = sum(yhat != y[:, 0])
    print(
        errors, "errors out of", len(y), "examples:",
        f"{100*errors/len(y)}% error rate"
    )


def generate_quadratic_data(
    size: int, seed: int = 1, scale: float = 0.7
) -> tuple[list[float]]:
    """
    Generates dataset based on the equation x^2 with some added noise.

    Args:
        size (int): Size of dataset.
        seed (int, optional): Random seed. Defaults to 1.
        scale (float, optional): Scale of the noise. Defaults to 0.7.

    Returns:
        tuple[list[float]]:
            Training values of x with some noise,
            training values of y with some noise,
            ideal values of y for the training values.
    """
    x_train = np.linspace(0, 49, size)
    np.random.seed(seed)
    y_ideal = x_train**2
    y_train = y_ideal + scale * y_ideal * (np.random.sample((size,)) - 0.5)
    return x_train, y_train, y_ideal


def plot_quadratic_noise_data(
    ideal: tuple[ndarray],
    train: tuple[ndarray],
    test: tuple[ndarray],
    cv: tuple[ndarray] = None,
    predict: tuple[ndarray] = None
) -> None:
    """
    Generates plot for the quadratic data with noise.

    Args:
        x (ndarray): X values.
        y_ideal (ndarray): Ideal y values.
        x_train (ndarray): X training values.
        y_train (ndarray): Y training values.
        x_test (ndarray): X test values.
        y_test (ndarray): Y test values.
    """
    _, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(ideal[0], ideal[1], "--", color="orangered", label="y_ideal", lw=1)
    title = "Training, Test"

    ax.scatter(train[0], train[1], color="red", label="train")
    ax.scatter(test[0], test[1], color="blue", label="test")

    if cv is not None:
        title += ", CV"
        ax.scatter(cv[0], cv[1], color="orange", label="cv")

    if predict is not None:
        ax.plot(predict[0], predict[1], color="green", label="predicted", lw=1)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc='upper left')


def plot_optimal_val(
    ideal: tuple[ndarray], train: tuple[ndarray], cv: tuple[ndarray],
    pred: tuple[ndarray], errors: tuple[ndarray], max_degree: int,
    optimal_val: int, lambdas: ndarray = None
):
    """
    Plots prediction line.

    Args:
        x (ndarray): X values.
        y (ndarray): Y values.
        color (str, optional): Color of the line. Defaults to "b".
    """
    comp = "Degree" if lambdas is None else "Lambda"
    _, ax = plt.subplots(2, 3, figsize=(12, 8))

    for i, degrees in enumerate(np.array_split(range(max_degree), 5)):
        row, col = i//3, i % 3
        ax[row, col].set_title("Predictions vs Data", fontsize=12)
        ax[row, col].set_xlabel("x")
        ax[row, col].set_ylabel("y")

        ax[row, col].plot(
            ideal[0], ideal[1], "--", color="purple", label="y_ideal", lw=1)
        ax[row, col].scatter(
            train[0], train[1], color="red", label="train", s=10)
        ax[row, col].scatter(
            cv[0], cv[1], color="green", label="cv", s=10)
        for i in degrees:
            ax[row, col].plot(
                pred[0], pred[1][:, i],  lw=2,
                label=f"{i+1}" if lambdas is None else f"{lambdas[i]}"
            )
        ax[row, col].legend(loc='upper left')

    x_vals = range(1, max_degree+1) if lambdas is None else lambdas
    ax[1, 2].plot(x_vals, errors[0], marker='o', label="Train error")
    ax[1, 2].plot(x_vals, errors[1], marker='o', label="CV error")
    if lambdas is not None:
        ax[1, 2].set_xscale('log')
        optimal_val = lambdas[optimal_val]
    ax[1, 2].axvline(optimal_val, color="purple")
    ax[1, 2].annotate(
        f"Optimal {comp}", xy=(optimal_val, 80000), xycoords='data',
        xytext=(0.25, 0.7), textcoords='axes fraction', fontsize=10,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    ax[1, 2].set_title(f"Error vs {comp}")
    ax[1, 2].legend(loc="upper center")
    ax[1, 2].set_xlabel(comp)
    ax[1, 2].set_ylabel("error")
    plt.tight_layout()
