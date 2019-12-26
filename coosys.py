import numpy as np

eps = 10**(-10)


def scalar_mul(vector1, vector2):
    """
    :param vector1: np.ndarray
    :param vector2: np.ndarray
    :return: float

    Scalar multiplication of vectors "vector1" and "vector2".
    """
    return np.sum(vector1 * vector2)


def vector_length(vector):
    """
    :param vector: np.ndarray
    :return: float

    Length of vector "vector".
    """
    return np.sum(vector**2)**0.5


def vector_norm(vector):
    """
    :param vector: np.ndarray
    :return: float

    Norm of vector "vector".
    """
    return np.sum(vector**2)


def vector_mul(vector1, vector2):
    """
    :param vector1: np.ndarray
    :param vector2: np.ndarray
    :return: np.ndarray

    Vector multiplication of vectors "vector1" and "vector2" (with shape of (3,) each).
    """
    return np.array(
        [vector1[1] * vector2[2] - vector1[2] * vector2[1],
        vector1[2] * vector2[0] - vector1[0] * vector2[2],
        vector1[0] * vector2[1] - vector1[1] * vector2[0]],
        dtype="float64"
    )


def vector_angle(vector1, vector2):
    """
    :param vector1: np.ndarray
    :param vector2: np.ndarray
    :return: float

    Angle between two vectors "vector1" and "vector2" without direction (i.e. smaller of two angles).
    """
    vector1 = vector1 / vector_length(vector1)
    vector2 = vector2 / vector_length(vector2)
    return np.arccos(scalar_mul(vector1, vector2))


def vector_angle_sign(vector1, vector2, axis):
    """
    :param vector1: np.ndarray
    :param vector2: np.ndarray
    :param axis: np.ndarray
    :return: float

    Angle between two vectors "vector1" and "vector2" with direction "axis" (all vectors are of shape (3,)).
    The angle could be thought of as anticlockwise angle between vectors regarding point of view on the end of axis.
    """
    direction = vector_mul(vector1, vector2)
    scalar = scalar_mul(direction, axis)
    if scalar > eps:
        return vector_angle(vector1, vector2)
    elif scalar < -eps:
        return 2 * np.pi - vector_angle(vector1, vector2)
    else:
        return 0.


def matrix_mul(matrix, vector):
    """
    :param matrix: np.ndarray
    :param vector: np.ndarray
    :return: np.ndarray

    Multiplication of matrix "matrix" (shape (n, n)) and vector "vector" (shape (n,)).
    """
    return matrix @ vector


def rotation_matrix(angle, axis):
    """
    :param angle: float
    :param axis: np.ndarray
    :return: np.ndarray

    Returns matrix of rotation in 3D space for rotation regarding axis "axis" for an angle "angle".
    """
    axis = axis / vector_length(axis)
    cosa = np.cos(angle)
    sina = np.sin(angle)
    matrix = np.array(
        [[cosa + (1 - cosa) * axis[0]**2, (1 - cosa) * axis[0] * axis[1] - sina * axis[2], (1 - cosa) * axis[0] * axis[2] + sina * axis[1]],
         [(1 - cosa) * axis[1] * axis[0] + sina * axis[2], cosa + (1 - cosa) * axis[1]**2, (1 - cosa) * axis[1] * axis[2] - sina * axis[0]],
         [(1 - cosa) * axis[2] * axis[0] - sina * axis[1], (1 - cosa) * axis[2] * axis[1] + sina * axis[0], cosa + (1 - cosa) * axis[2]**2]],
        dtype="float64")
    return matrix


def rotate_vector(vector, angle, axis):
    """
    :param vector: np.ndarray
    :param angle: float
    :param axis: np.ndarray
    :return: np.ndarray

    Rotates vector "vector" regarding axis "axis" for an angle "angle".
    """
    matrix = rotation_matrix(angle, axis)
    return matrix @ vector


def cartesian_to_kepler(data, gamma=398603*10**9):
    """
    :param data: np.ndarray
    :param gamma: float
    :return: np.ndarray

     Receives numpy array of cartesian coordinates with shape (6,) as [x, y, z, vx, vy, vz] (and, optionally,
     parameter gamma = GM for celestial body; gamma for Earth is set by default) and returns numpy array of
     orbit elements with shape (6,) as [a, e, inclination, longitude, argument, anomaly] where "a" is semimajor axis,
     "e" is eccentricity, "inclination" is an angle between plane Oxy and plane of orbit, "longitude" is an angle
     between Ox and ascending node (regarding Oz), "argument" is an angle between ascending node and pericenter
     (regarding vector of angle velocity) and "anomaly" is an angle between pericenter and current position of body
     (regarding vector of angle velocity).

     Potential issues:
     1. Longitude is not defined in case of zero inclination. this case is processed as zero longitude.
     Correspondingly, longitude could be unstable for small inclinations.
     2. Argument is not defined for case of zero eccentricity, this case is processed as zero argument.
     Correspondingly, argument could be unstable for small eccentricities. Thus, similar problem occurs for anomaly.
    """
    r = data[:3]
    v = data[3:]

    E = vector_norm(v) / 2 - gamma / vector_length(r)
    assert E < -eps, "Body is escaping gravity"

    L = vector_mul(r, v)
    w = L / vector_norm(r)  # angular velocity vector
    l = L / vector_length(L)

    p = vector_norm(L) / gamma  # parameter of orbit
    e2 = 1 + 2 * E * vector_norm(L) / gamma**2
    if np.abs(e2) < eps:  # orbit is close to circular
        e = 0.
    else:
        e = e2**0.5
    a = - gamma / (2 * E)

    inclination = vector_angle(l, np.array([0., 0., 1.], dtype="float64"))

    lxy = np.array([l[0], l[1], 0.], dtype="float64")

    if np.sum(np.abs(lxy - np.array([0., 0., 0.], dtype="float64"))) < eps:  # Orbit is in Oxy
        longitude = 0.
    else:
        longitude = vector_angle_sign(np.array([0., -1., 0.], dtype="float64"), lxy, np.array([0., 0., 1.], dtype="float64"))

    ascending = rotate_vector(np.array([1., 0., 0.], dtype="float64"), longitude, np.array([0., 0., 1.], dtype="float64"))

    if e == 0.:  # Orbit is circular
        argument = 0.
        anomaly = vector_angle_sign(ascending, r, l)
    else:
        cosa = (p / vector_length(r) - 1) / 3
        sina = scalar_mul(v - vector_mul(w, r), r) * p / (vector_length(r) ** 3 * e)
        anomaly = np.arctan2(sina, cosa)

        pericenter = rotate_vector(r, -anomaly, l)
        pericenter = pericenter / vector_length(pericenter)

        argument = vector_angle_sign(ascending, pericenter, l)

    return np.array([a, e, inclination, longitude, argument, anomaly], dtype="float64")


def kepler_to_cartesian(data, gamma=398603*10**9):
    """
    :param data: np.ndarray
    :param gamma: float
    :return: np.ndarray

     Receives numpy array of orbital elements with shape (6,) as [a, e, inclination, longitude, argument, anomaly]
     (and, optionally, parameter gamma = GM for celestial body; gamma for Earth is set by default)
     and returns numpy array of cartesian coordinates with shape (6,) as [x, y, z, vx, vy, vz].

     The function is reverse for "cartesian_to_kepler".
    """
    a, e, inclination, longitude, argument, anomaly = data

    ascending = rotate_vector(np.array([1., 0., 0.], dtype="float64"), longitude, np.array([0., 0., 1.], dtype="float64"))

    axis = rotate_vector(ascending, np.pi / 2, np.array([0., 0., 1.], dtype="float64"))
    axis = rotate_vector(axis, inclination, ascending)

    l = vector_mul(ascending, axis)
    p = a * (1 - e**2)
    L = l * (p * gamma)**0.5

    r = rotate_vector(ascending, argument + anomaly, l)
    r *= p / (1 + e * np.cos(anomaly))

    f = L / vector_norm(r)
    v = vector_mul(f, r) + r * vector_length(r) * e / p * np.sin(anomaly)

    return np.concatenate([r, v])



