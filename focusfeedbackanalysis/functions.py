from __future__ import annotations

import numpy as np
from numdifftools import Hessian

from . import utilities


def linefitdt(x, y):
    # usage: d, t, G2 = linefitdt(x,y)
    #
    # Fits a line x*sin(t)-y*cos(t)=d through x,y using least squares fitting
    # with the perpendicular distance from the points to the line. In this way
    # the fit result is independent on which variable is the dependent variable.
    #
    # Complicated math is involved, look at Wim's notes...

    # Remove nan's
    x, y = utilities.rmnan(x, y)
    x = np.array(x)
    y = np.array(y)

    if x.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Prepare some numbers
    mx = np.mean(x)
    my = np.mean(y)
    cv = np.cov(x, y, bias=True)

    # The actual algebra
    z = cv.diagonal().dot((-1, 1)) / np.mean(cv.sum(1) - cv.diagonal())
    if np.any(np.isinf(z)):
        d = (mx, my)
        t = (np.pi / 4, 0)
    else:
        p = 0.5 * (np.array((1, -1)) * np.sqrt(z**2 + 4) - z)
        t = np.arctan(1 / p)
        d = mx / p / np.sqrt(1 / p**2 + 1) - my / np.sqrt((p**2 + 1) / p**2)

    # Remove anomalous results
    d, t = utilities.rmnan(d, t)

    # Get the right set of parameters
    g = lambda f: np.sqrt(np.sum((y * np.cos(f[1]) - x * np.sin(f[1]) + f[0]) ** 2))
    g2 = [g(i) / (len(x) - 2) for i in zip(d, t)]
    i = np.argmin(g2)
    d = d[i]
    t = t[i]
    g2 = g2[i]
    if d < 0:
        d *= -1
        t += np.pi
    elif np.isnan(d):
        d = np.linalg.norm((mx, my))
        t = np.nan
        g2 = np.nan
    t %= 2 * np.pi

    dt, dd = np.sqrt(g2) * np.sqrt(np.diag(np.linalg.inv(Hessian(g)((d, t)))))

    return d, t, g2, dd, dt
