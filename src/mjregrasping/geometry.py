import matplotlib.pyplot as plt
import numpy as np


def point_to_line_segment(p1, p2, p3):
    """ https://paulbourke.net/geometry/pointlineplane/ """
    d = p2 - p1  # delta
    u = np.dot(d, (p3 - p1)) / np.sum(np.square(d))  # project

    # clip to the ends of the segment
    nearest_point = p1 + np.clip(u, 0, 1) * d

    return nearest_point, u


def pairwise_squared_distances(a, b):
    """
    Adapted from https://github.com/ClayFlannigan/icp
    Computes pairwise distances between to sets of points

    Args:
        a: [b, ..., n, k]
        b:  [b, ..., m, k]

    Returns: [b, ..., n, m]

    """
    a_s = np.sum(np.square(a), axis=-1, keepdims=True)  # [b, ..., n, 1]
    b_s = np.sum(np.square(b), axis=-1, keepdims=True)  # [b, ..., m, 1]
    dist = a_s - 2 * a @ np.moveaxis(b, -1, -2) + np.moveaxis(b_s, -1, -2)  # [b, ..., n, m]
    return dist


def squared_norm(x, **kwargs):
    return np.sum(np.square(x), axis=-1, **kwargs)


def main():
    p1 = np.array([0, 0])
    p2 = np.array([1, 0.2])
    p3 = np.array([0.5, -0.1])
    nearest_point, _ = point_to_line_segment(p1, p2, p3)

    plt.axis("equal")
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
    plt.plot([p3[0], nearest_point[0]], [p3[1], nearest_point[1]], 'r-')
    plt.plot(p1[0], p1[1], 'ko')
    plt.plot(p2[0], p2[1], 'ko')
    plt.plot(p3[0], p3[1], 'ro')
    plt.plot(nearest_point[0], nearest_point[1], 'ro')
    plt.show()


if __name__ == '__main__':
    main()
