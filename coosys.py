import numpy as np

eps = 10**(-10)

Ox = np.array([1, 0, 0], dtype="float64")
Oy = np.array([0, 1, 0], dtype="float64")
Oz = np.array([0, 0, 1], dtype="float64")


def scalar_mul(vector1, vector2):
    """
    :param vector1: np.ndarray
    :param vector2: np.ndarray
    :return: float

    Scalar multiplication of vectors "vector1" and "vector2".
    """
    return np.sum(vector1 * vector2)


def vector_norm(vector):
    """
    :param vector: np.ndarray
    :return: float

    Norm of vector "vector".
    """
    return np.sum(vector**2)


def vector_length(vector):
    """
    :param vector: np.ndarray
    :return: float

    Length of vector "vector".
    """
    return vector_norm(vector)**0.5


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
     orbital elements with shape (6,) as [a, e, inclination, longitude, argument, anomaly] where "a" is semimajor axis,
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

    inclination = vector_angle(l, Oz)

    lxy = np.array([l[0], l[1], 0.], dtype="float64")

    if np.sum(np.abs(lxy)) < eps:  # Orbit is in Oxy
        longitude = 0.
    else:
        longitude = vector_angle_sign(-Oy, lxy, Oz)

    ascending = rotate_vector(Ox, longitude, Oz)

    if e == 0.:  # Orbit is circular
        argument = 0.
        anomaly = vector_angle_sign(ascending, r, l)
    else:
        cosa = np.clip((p / vector_length(r) - 1) / e, -1., 1.)
        sign = scalar_mul(v, r)
        anomaly = np.arccos(cosa)
        if sign < -eps:
            anomaly = 2 * np.pi - anomaly

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

    ascending = rotate_vector(Ox, longitude, Oz)

    axis = rotate_vector(ascending, np.pi / 2, Oz)
    axis = rotate_vector(axis, inclination, ascending)

    l = vector_mul(ascending, axis)
    p = a * (1 - e**2)
    L = l * (p * gamma)**0.5

    r = rotate_vector(ascending, argument + anomaly, l)
    r *= p / (1 + e * np.cos(anomaly))

    w = L / vector_norm(r)
    v = vector_mul(w, r) + r * vector_length(r) * e / p * np.sin(anomaly)

    return np.concatenate([r, v])


def quaternion(u, v):
    """
    :param u: float
    :param v: np.ndarray
    :return: np.ndarray

    Returns quaternion with scalar part of "u" and vector part of "v"
    """
    return np.concatenate([np.array([u], dtype="float64"), v])


def quaternion_from_angle(angle, v):
    """
    :param angle: float
    :param v: np.ndarray
    :return: np.ndarray

    Returns quaternion of rotation for angle "angle" around axis "v"
    """
    return np.concatenate([np.array([np.cos(angle)], dtype="float64"), v * np.sin(angle)])


def quaternion_norm(q):
    """
    :param q: np.ndarray
    :return: float

    Norm of quaternion "q".
    """
    return np.sum(q**2)


def quaternion_length(q):
    """
    :param q: np.ndarray
    :return: float

    Length of quaternion "q".
    """
    return quaternion_norm(q)**0.5


def quaternion_multiplication(q1, q2):
    """
    :param q1: np.ndarray
    :param q2: np.ndarray
    :return: np.ndarray

    Quaternion multiplication of quaternions "q1" and "q2" (with shape of (4,) each).
    """
    u1, v1 = q1[0], q1[1:]
    u2, v2 = q2[0], q2[1:]
    u = u1 * u2 - scalar_mul(v1, v2)
    v = u1 * v2 + u2 * v1 + vector_mul(v1, v2)
    return quaternion(u, v)


def reverse_quaternion(q):
    """
    :param q: np.ndarray
    :return: np.ndarray

    Returns quaternion reverse to "q" (both are of shape (4,))
    """
    return quaternion(q[0], -q[1:]) / quaternion_norm(q)


def apply_quaternion(x, q):
    """
    :param x: np.ndarray
    :param q: np.ndarray
    :return: np.ndarray

    Applies quaternion-based rotation to vector "x" (of shape (3,)) with quaternion "q" (of shape (4,)) and returns
    vector of shape (3,)
    """
    return quaternion_multiplication(quaternion_multiplication(q, quaternion(0, x)), reverse_quaternion(q))[1:]


def cartesian_to_quaternion(data, gamma=398603*10**9):
    """
    :param data: np.ndarray
    :param gamma: float
    :return: np.ndarray

     Receives numpy array of cartesian coordinates with shape (6,) as [x, y, z, vx, vy, vz] (and, optionally,
     parameter gamma = GM for celestial body; gamma for Earth is set by default) and returns numpy array of
     orbital elements with shape (7,) as [a, e, anomaly] + q where "a" is semimajor axis, "e" is eccentricity,
     "anomaly" is an angle between pericenter and current position of body (regarding vector of angle velocity)
     and "q" is quaternion of rotating from Oxyz to orbit's support basis.

     Potential issues:
     1. Pericenter is not defined for case of zero eccentricity, in this case ascending node is chosen as pericenter.
     Correspondingly, pericenter direction could be unstable for small eccentricities.
     Thus, similar problem occurs for anomaly.
    """
    a, e, inclination, longitude, argument, anomaly = cartesian_to_kepler(data, gamma)

    q = quaternion_from_angle(longitude / 2, Oz)

    ascending = apply_quaternion(Ox, q)
    q = quaternion_multiplication(quaternion_from_angle(inclination / 2, ascending), q)

    l = vector_mul(data[:3], data[3:])
    q = quaternion_multiplication(quaternion_from_angle(argument / 2, l), q)

    q /= quaternion_length(q)

    return np.concatenate([[a, e, anomaly], q])


def quaternion_to_cartesian(data, gamma=398603*10**9):
    """
    :param data: np.ndarray
    :param gamma: float
    :return: np.ndarray

     Receives numpy array of orbital elements with shape (7,) as [a, e, anomaly] + q
     (and, optionally, parameter gamma = GM for celestial body; gamma for Earth is set by default)
     and returns numpy array of cartesian coordinates with shape (6,) as [x, y, z, vx, vy, vz].

     The function is reverse for "cartesian_to_quaternion".
    """
    a, e, anomaly = data[:3]
    q = data[3:]

    p = a * (1 - e ** 2)

    pericenter = apply_quaternion(Ox, q)
    l = apply_quaternion(Oz, q)
    L = l * (p * gamma)**0.5

    r = rotate_vector(pericenter, anomaly, l)
    r *= p / (1 + e * np.cos(anomaly))

    w = L / vector_norm(r)
    v = vector_mul(w, r) + r * vector_length(r) * e / p * np.sin(anomaly)

    return np.concatenate([r, v])

