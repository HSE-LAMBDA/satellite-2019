import numpy as np

eps = 10**(-10)


def scalar_mul(vector1, vector2):
    return np.sum(vector1 * vector2)


def vector_length(vector):
    return np.sum(vector**2)**0.5


def vector_norm(vector):
    return np.sum(vector**2)


def vector_mul(vector1, vector2):
    return np.array(
        [vector1[1] * vector2[2] - vector1[2] * vector2[1],
        vector1[2] * vector2[0] - vector1[0] * vector2[2],
        vector1[0] * vector2[1] - vector1[1] * vector2[0]],
        dtype="float64"
    )


def vector_angle(vector1, vector2):
    vector1 = vector1 / vector_length(vector1)
    vector2 = vector2 / vector_length(vector2)
    return np.arccos(scalar_mul(vector1, vector2))


def vector_angle_sign(vector1, vector2, axis):
    direction = vector_mul(vector1, vector2)
    scalar = scalar_mul(direction, axis)
    if scalar > eps:
        return vector_angle(vector1, vector2)
    elif scalar < eps:
        return 2 * np.pi - vector_angle(vector1, vector2)
    else:
        return 0.


def matrix_mul(matrix, vector):
    return matrix @ vector


def turn_matrix(angle, axis):
    axis = axis / vector_length(axis)
    cosa = np.cos(angle)
    sina = np.sin(angle)
    matrix = np.array(
        [[cosa + (1 - cosa) * axis[0]**2, (1 - cosa) * axis[0] * axis[1] - sina * axis[2], (1 - cosa) * axis[0] * axis[2] + sina * axis[1]],
         [(1 - cosa) * axis[1] * axis[0] + sina * axis[2], cosa + (1 - cosa) * axis[1]**2, (1 - cosa) * axis[1] * axis[2] - sina * axis[0]],
         [(1 - cosa) * axis[2] * axis[0] - sina * axis[1], (1 - cosa) * axis[2] * axis[1] + sina * axis[0], cosa + (1 - cosa) * axis[2]**2]],
        dtype="float64")
    return matrix


def turn_vector(vector, angle, axis):
    matrix = turn_matrix(angle, axis)
    return matrix @ vector


def cartesian_to_kepler(data, gamma=398603*10**9):
    r = data[:3]
    v = data[3:]
    E = vector_norm(v) / 2 - gamma / vector_length(r)
    assert E < 0, "Body is escaping gravity"
    L = vector_mul(r, v)
    f = L / vector_norm(r)
    l = L / vector_length(L)
    p = vector_norm(L) / gamma
    e = (1 + 2 * E * vector_norm(L) / gamma**2)**0.5
    a = - gamma / (2 * E)

    cosa = (p / vector_length(r) - 1) / 3
    sina = scalar_mul(v - vector_mul(f, r), r) * p / (vector_length(r) ** 3 * e)
    anomaly = np.arctan2(sina, cosa)

    inclination = vector_angle(l, np.array([0., 0., 1.], dtype="float64"))

    lxy = np.array([l[0], l[1], 0.], dtype="float64")

    if np.sum(np.abs(lxy - np.array([0., 0., 0.], dtype="float64"))) < eps:
        longitude = 0.
    else:
        longitude = vector_angle_sign(np.array([0., -1., 0.], dtype="float64"), lxy, np.array([0., 0., 1.], dtype="float64"))

    rising = turn_vector(np.array([1., 0., 0.], dtype="float64"), longitude, np.array([0., 0., 1.], dtype="float64"))

    pericenter = turn_vector(r, -anomaly, l)
    pericenter = pericenter / vector_length(pericenter)

    argument = vector_angle_sign(rising, pericenter, l)

    return np.array([a, e, inclination, longitude, argument, anomaly], dtype="float64")


def kepler_to_cartesian(data, gamma=398603*10**9):
    a, e, inclination, longitude, argument, anomaly = data
    rising = turn_vector(np.array([1., 0., 0.], dtype="float64"), longitude, np.array([0., 0., 1.], dtype="float64"))
    axis = turn_vector(rising, np.pi / 2, np.array([0., 0., 1.], dtype="float64"))
    axis = turn_vector(axis, inclination, rising)
    l = vector_mul(rising, axis)
    r = turn_vector(rising, argument + anomaly, l)
    p = a * (1 - e**2)
    L = l * (p * gamma)**0.5
    r *= p / (1 + e * np.cos(anomaly))
    f = L / vector_norm(r)
    v = vector_mul(f, r) + r * vector_length(r) * e / p * np.sin(anomaly)
    return np.concatenate([r, v])



