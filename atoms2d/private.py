import numpy as np


class Polygon:
    def __init__(self, sides, lat_const):
        self.center = [0, 0]
        self.coords = []
        self.sides = sides
        self.length = lat_const * np.sqrt(3) / (4 * np.cos(np.pi / (2 * sides))
                                                * np.cos(np.pi / sides))
        self.radius = self.length / (2 * np.sin(np.pi / sides))

        for i in range(sides):
            ratio = float(i) / sides
            self.coords.append([
                np.sin(-ratio * 2 * np.pi) * self.radius,
                np.cos(ratio * 2 * np.pi) * self.radius,
            ])

    def translate(self, vector):
        self.center[0] += vector[0]
        self.center[1] += vector[1]

        for i in range(len(self.coords)):
            self.coords[i][0] += vector[0]
            self.coords[i][1] += vector[1]

    def rotate(self, angle, center=None):
        if center is None:
            center = self.center

        for i in range(len(self.coords)):
            x = self.coords[i][0] - center[0]
            y = self.coords[i][1] - center[1]

            self.coords[i][0] = ((x * np.cos(angle)) - (y * np.sin(angle))
                                 + center[0])
            self.coords[i][1] = ((x * np.sin(angle)) + (y * np.cos(angle))
                                 + center[1])


def where(array, equal):
    """Find the index of a 2D array where it equals to another array."""
    for i, a in enumerate(array):
        if (a == equal).all():
            return i


def is_even(integer):
    """Return true if number is even, false otherwise."""
    return integer % 2 == 0


def is_tm(symbol):
    """Return true if an element is a transition metal."""
    return symbol in ['Mo', 'W']


def is_dc(symbol):
    """Return true if an element is a chalcogen."""
    return symbol in ['S', 'Se']


def is_same_type(a, b):
    """Return true if both material belong to the same group."""
    return is_tm(a) == is_tm(b) or is_dc(a) == is_dc(b)


def is_same_plane(a, b):
    """Return true if both position lie roughly on the same plane."""
    return abs(a[2] - b[2]) < 1
