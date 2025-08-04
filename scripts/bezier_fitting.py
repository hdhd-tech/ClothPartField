from typing import Union, List, Tuple
import numpy as np
from scipy.optimize import minimize


def bezier_curve(
    t: Union[float, np.ndarray],
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> np.ndarray:
    """
    Calculate points on a cubic Bézier curve.

    Args:
        t: Parameter values (0 to 1)
        p0: Start point (x, y)
        p1: First control point (x, y)
        p2: Second control point (x, y)
        p3: End point (x, y)

    Returns:
        Points on the Bézier curve
    """
    # Ensure t is a numpy array for vectorized operations
    t = np.asarray(t)

    # Cubic Bézier formula: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
    return (
        (1 - t) ** 3 * p0[:, np.newaxis]
        + 3 * (1 - t) ** 2 * t * p1[:, np.newaxis]
        + 3 * (1 - t) * t**2 * p2[:, np.newaxis]
        + t**3 * p3[:, np.newaxis]
    )


def fit_bezier_curve(
    points: List[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a cubic Bézier curve to a sequence of points.

    Args:
        points: List of (x, y) tuples representing the data points
        num_samples: Number of samples for curve evaluation during fitting

    Returns:
        Tuple of (p0, p1, p2, p3) representing the four control points
    """
    # Convert points to numpy array
    data_points = np.array(points)

    # Start and end points are fixed
    p0 = data_points[0]
    p3 = data_points[-1]

    # Initial guess for control points (simple linear interpolation)
    p1_init = p0 + (p3 - p0) / 3
    p2_init = p0 + 2 * (p3 - p0) / 3

    # Combine initial control points into a single parameter vector
    initial_params = np.concatenate([p1_init, p2_init])

    def objective_function(params):
        """Objective function to minimize - sum of squared distances."""
        # Extract control points from parameter vector
        p1 = params[:2]
        p2 = params[2:]

        # Generate parameter values for data points (uniform spacing assumption)
        t_values = np.linspace(0, 1, len(data_points))

        # Calculate curve points
        curve_points = bezier_curve(t_values, p0, p1, p2, p3)

        # Calculate squared distance between curve and data points
        diff = curve_points.T - data_points
        return np.sum(diff**2)

    # Optimize to find best control points
    result = minimize(objective_function, initial_params, method="BFGS")

    # Extract optimized control points
    p1_opt = result.x[:2]
    p2_opt = result.x[2:]

    return p0, p1_opt, p2_opt, p3
