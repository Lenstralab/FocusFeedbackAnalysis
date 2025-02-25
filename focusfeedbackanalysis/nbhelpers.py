from __future__ import annotations

from typing import Sequence

import numpy as np
from numba import jit


@jit(nopython=True, nogil=True)
def meshgrid2(x: Sequence[int], y: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    s = (len(y), len(x))
    xv = np.zeros(s)
    yv = np.zeros(s)
    for i in range(s[0]):
        for j in range(s[1]):
            xv[i, j] = x[j]
            yv[i, j] = y[i]
    return xv, yv


@jit(nopython=True, nogil=True)
def meshgrid3(x: Sequence[int], y: Sequence[int], z: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = (len(y), len(x), len(z))
    xv = np.zeros(s)
    yv = np.zeros(s)
    zv = np.zeros(s)
    for i in range(s[0]):
        for j in range(s[1]):
            for k in range(s[2]):
                xv[i, j, k] = x[j]
                yv[i, j, k] = y[i]
                zv[i, j, k] = z[k]
    return xv, yv, zv


@jit(nopython=True, nogil=True)
def erf(x: float) -> float:
    # save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y  # erf(-x) = -erf(x)


@jit(nopython=True, nogil=True)
def erf2(x: np.ndarray) -> np.ndarray:
    s = x.shape
    y = np.zeros(s)
    for i in range(s[0]):
        for j in range(s[1]):
            y[i, j] = erf(x[i, j])  # type: ignore
    return y


@jit(nopython=True, nogil=True)
def erf3(x: np.ndarray) -> np.ndarray:
    s = x.shape
    y = np.zeros(s)
    for i in range(s[0]):
        for j in range(s[1]):
            for k in range(s[2]):
                y[i, j, k] = erf(x[i, j, k])  # type: ignore
    return y


@jit(nopython=True, nogil=True)
def gaussian5(p: Sequence[float], size_x: int, size_y: int) -> np.ndarray:
    """p: [x,y,fwhm,area,offset]
    X,Y: size of image
    reimplemented for numba, small deviations from true result
        possible because of reimplementation of erf
    """
    if p[2] == 0:
        efac = 1e-9
    else:
        efac = np.sqrt(np.log(2)) / p[2]
    dx = efac
    dy = efac
    xv, yv = meshgrid2(np.arange(size_y) - p[0], np.arange(size_x) - p[1])
    x = 2 * dx * xv
    y = 2 * dy * yv
    return p[3] / 4 * (erf2(x + dx) - erf2(x - dx)) * (erf2(y + dy) - erf2(y - dy)) + p[4]


@jit(nopython=True, nogil=True)
def gaussian6(p: Sequence[float], size_x: int, size_y: int) -> np.ndarray:
    """p: [x,y,fwhm,area,offset,ellipticity]
    X,Y: size of image
    reimplemented for numba, small deviations from true result
        possible because of reimplementation of erf
    """
    if p[2] == 0:
        efac = 1e-9
    else:
        efac = np.sqrt(np.log(2)) / p[2]
    dx = efac / p[5]
    dy = efac * p[5]
    xv, yv = meshgrid2(np.arange(size_y) - p[0], np.arange(size_x) - p[1])
    x = 2 * dx * xv
    y = 2 * dy * yv
    return p[3] / 4 * (erf2(x + dx) - erf2(x - dx)) * (erf2(y + dy) - erf2(y - dy)) + p[4]


@jit(nopython=True, nogil=True)
def gaussian7(p: Sequence[float], size_x: int, size_y: int) -> np.ndarray:
    """p: [x,y,fwhm,area,offset,ellipticity,angle towards x-axis]
    X,Y: size of image
    reimplemented for numba, small deviations from true result
        possible because of reimplementation of erf
    """
    if p[2] == 0:
        efac = 1e-9
    else:
        efac = np.sqrt(np.log(2)) / p[2]
    dx = efac / p[5]
    dy = efac * p[5]
    xv, yv = meshgrid2(np.arange(size_y) - p[0], np.arange(size_x) - p[1])
    cos, sin = np.cos(p[6]), np.sin(p[6])
    x = 2 * dx * (cos * xv - yv * sin)
    y = 2 * dy * (cos * yv + xv * sin)
    return p[3] / 4 * (erf2(x + dx) - erf2(x - dx)) * (erf2(y + dy) - erf2(y - dy)) + p[4]


@jit(nopython=True, nogil=True)
def gaussian9(p: Sequence[float], size_x: int, size_y: int) -> np.ndarray:
    """p: [x,y,fwhm,area,offset,ellipticity,angle towards x-axis,tilt-x,tilt-y]
    X,Y: size of image
    reimplemented for numba, small deviations from true result
        possible because of reimplementation of erf
    """
    if p[2] == 0:
        efac = 1e-9
    else:
        efac = np.sqrt(np.log(2)) / p[2]
    dx = efac / p[5]
    dy = efac * p[5]
    xv, yv = meshgrid2(np.arange(size_y) - p[0], np.arange(size_x) - p[1])
    cos, sin = np.cos(p[6]), np.sin(p[6])
    x = 2 * dx * (cos * xv - yv * sin)
    y = 2 * dy * (cos * yv + xv * sin)
    return p[3] / 4 * (erf2(x + dx) - erf2(x - dx)) * (erf2(y + dy) - erf2(y - dy)) + p[4] + p[7] * xv + p[8] * yv


@jit(nopython=True, nogil=True)
def gaussian10(p: Sequence[float], size_x: int, size_y: int, size_z: int) -> np.ndarray:
    return gaussian10grid(p, *meshgrid3(np.arange(size_x), np.arange(size_y), np.arange(size_z)))


@jit(nopython=True, nogil=True)
def gaussian7grid(p: Sequence[float], xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
    """p: [x,y,fwhm,area,offset,ellipticity,angle towards x-axis]
    xv, yv = meshgrid(np.arange(Y),np.arange(X))
        calculation of meshgrid is done outside, so it doesn't
        have to be done each time this function is run
    reimplemented for numba, small deviations from true result
        possible because of reimplementation of erf
    """
    if p[2] == 0:
        efac = 1e-9
    else:
        efac = np.sqrt(np.log(2)) / p[2]
    dx = efac / p[5]
    dy = efac * p[5]
    cos, sin = np.cos(p[6]), np.sin(p[6])
    x = 2 * dx * (cos * (xv - p[0]) - (yv - p[1]) * sin)
    y = 2 * dy * (cos * (yv - p[1]) + (xv - p[0]) * sin)
    return p[3] / 4 * (erf2(x + dx) - erf2(x - dx)) * (erf2(y + dy) - erf2(y - dy)) + p[4]


@jit(nopython=True, nogil=True)
def gaussian9grid(p: Sequence[float], xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
    """p: [x,y,fwhm,area,offset,ellipticity,angle towards x-axis,tilt-x,tilt-y]
    xv, yv = meshgrid(np.arange(Y),np.arange(X))
        calculation of meshgrid is done outside, so it doesn't
        have to be done each time this function is run
    reimplemented for numba, small deviations from true result
        possible because of reimplementation of erf
    """
    if p[2] == 0:
        efac = 1e-9
    else:
        efac = np.sqrt(np.log(2)) / p[2]
    dx = efac / p[5]
    dy = efac * p[5]
    cos, sin = np.cos(p[6]), np.sin(p[6])
    x = 2 * dx * (cos * (xv - p[0]) - (yv - p[1]) * sin)
    y = 2 * dy * (cos * (yv - p[1]) + (xv - p[0]) * sin)
    return (
        p[3] / 4 * (erf2(x + dx) - erf2(x - dx)) * (erf2(y + dy) - erf2(y - dy))
        + p[4]
        + p[7] * xv
        + p[8] * yv
        - p[7] * p[0]
        - p[8] * p[1]
    )


@jit(nopython=True, nogil=True)
def gaussian10grid(p: Sequence[float], xv: np.ndarray, yv: np.ndarray, zv: np.ndarray) -> np.ndarray:
    """p: [x, y, z, fwhm_xy, fwhm_z, area, offset, tilt-x, tilt-y, tilt-z]
    xv, yv, zv = meshgrid(np.arange(Y), np.arange(X), np.arange(Z))
        calculation of meshgrid is done outside, so it doesn't
        have to be done each time this function is run
    reimplemented for numba, small deviations from true result
        possible because of reimplementation of erf
    """
    dxy = 1e-9 if p[3] == 0 else np.sqrt(np.log(2)) / p[3]
    dz = 1e-9 if p[4] == 0 else np.sqrt(np.log(2)) / p[4]
    x = 2 * dxy * (xv - p[0])
    y = 2 * dxy * (yv - p[1])
    z = 2 * dz * (zv - p[2])
    return (
        p[5] / 8 * (erf3(x + dxy) - erf3(x - dxy)) * (erf3(y + dxy) - erf3(y - dxy)) * (erf3(z + dz) - erf3(z - dz))
        + p[6]
        + p[7] * xv
        + p[8] * yv
        + p[9] * zv
        - p[7] * p[0]
        - p[8] * p[1]
        - p[9] * p[2]
    )
