from __future__ import annotations

import warnings

import colorcet
import numpy as np
from jedi.inference.value.iterable import Sequence
from matplotlib.colors import ListedColormap
from numpy.typing import ArrayLike
from scipy import interpolate, optimize, special
from matplotlib import pyplot as plt


def p_(diff_c_t: ArrayLike, r: float, dr: float) -> ArrayLike:
    """ Probability of particle being in focus (moved max z from focus) after time t
        P in equation 5 in 10.1016/j.molcel.2024.01.020
    """
    return (
        special.erf(np.abs(r - dr) / 2 / np.sqrt(diff_c_t)) + special.erf(np.abs(r + dr) / 2 / np.sqrt(diff_c_t))
    ) / 2


# def p_in_focus0(diff_c: ArrayLike, r: float, dr: float, t: ArrayLike, dt: ArrayLike) -> ArrayLike:
#     return np.exp((t // dt) * np.log(p_(diff_c * t, r, dr)))
#
#
# def p_in_focus1(diff_c: ArrayLike, r: float, dr: float, t: ArrayLike, dt: ArrayLike) -> ArrayLike:
#     return p_(diff_c * t, r, dr) ** (t // dt)


def p_in_focus0(diff_c: ArrayLike, r: float, dr: float, dt: ArrayLike,
                max_t: float = 1e8, min_p: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """ Probability of particle being in focus (moved max z from focus) in every frame, no focus feedback
            P in equation 5 in 10.1016/j.molcel.2024.01.020
        """
    q = 1
    p = []
    t = []
    i = 1
    ti = 0
    while q > min_p and ti < max_t:
        ti = dt * i
        q *= p_(diff_c * dt * i, r, dr)
        t.append(ti)
        p.append(q)
        i += 1
    return np.array(t), np.array(p)


def p_in_focus1(diff_c: ArrayLike, r: float, dr: float, dt: ArrayLike,
                max_t: float = 1e8, min_p: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    # Probability particle in focus every frame up to time t with focus feedback
    q = 1
    p = []
    t = []
    i = 1
    p0 = p_(diff_c * dt, r, dr)
    ti = 0
    while q > min_p and ti < max_t:
        ti = dt * i
        q = p0 ** i
        t.append(ti)
        p.append(q)
        i += 1
    return np.array(t), np.array(p)


def total_time(diff_c: ArrayLike, r: ArrayLike, dr: ArrayLike, dt: ArrayLike, p: ArrayLike) -> ArrayLike:
    """ total time in focus
        T in equation 5 in 10.1016/j.molcel.2024.01.020
    """
    # abs to prevent going to -inf
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.abs(dt * np.log(p) / np.log(p_(diff_c * dt, r, dr)))


def get_dt(t: float, diff_c: float, r: float, dr: float, p: float) -> float:
    """ maximum time interval for given diffusion constant (um2/s), feedback range (um), feedback error (um) and p value
    """
    assert p < 1, "p value needs to be smaller than 1"
    log_t = np.log(t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        fit = optimize.minimize(lambda dt: np.log(total_time(diff_c, r, dr, dt, p)), r**2 / np.pi / diff_c)  # type: ignore
        assert fit.success
        minimum = np.log(fit.x[0])
        if t < total_time(diff_c, r, dr, fit.x[0], p):
            return np.inf

        x = np.logspace(minimum - 5, minimum + 5)
        y = np.log(total_time(diff_c, r, dr, x, p))
        idx = np.isfinite(y)
        f = interpolate.interp1d(x[idx], y[idx], fill_value="extrapolate")
        fit = optimize.minimize_scalar(lambda dt: (f(dt) - log_t) ** 2, (minimum - 10, minimum))
        assert fit.success
        fit = optimize.minimize(lambda dt: (total_time(diff_c, r, dr, dt, p) - t) ** 2, fit.x, method="Nelder-Mead")
        assert fit.success
    return fit.x[0]


def get_diff_c(t: float, dt: float, r: float, dr: float, p: float) -> float:
    """ maximum diffusion constant (um2/s) for given time interval (s),
        feedback range (um), feedback error (um) and p value
    """
    assert p < 1, "p value needs to be smaller than 1"
    log_t = np.log(t)
    with warnings.catch_warnings():
        x = np.logspace(-5, 5)
        y = np.log(total_time(x, r, dr, dt, p))
        idx = np.isfinite(y)
        f = interpolate.interp1d(x[idx], y[idx], fill_value="extrapolate")
        fit = optimize.minimize_scalar(lambda diff_c: (f(diff_c) - log_t) ** 2, (-10, 0))
        assert fit.success
        fit = optimize.minimize(lambda diff_c: (total_time(diff_c, r, dr, dt, p) - t) ** 2, fit.x, method="Nelder-Mead")
        assert fit.success
    return fit.x[0]


def plot_time_timeint(diff_c: float, r: float, dr: float, dt_range: Sequence[float, float] = (3, 25),
                      total_time_range: Sequence[float, float] = (1e0, 1e6)) -> None:
    cmap = ListedColormap(colorcet.diverging_gkr_60_10_c40)
    dt = np.linspace(*dt_range)  # type: ignore
    cs = 0.999, 0.99, 0.95, 0.75, 0.5, 0.25
    for i in cs:
        plt.semilogy(dt, total_time(diff_c, r, dr, dt, i), label=f"p: {i}", color=cmap(int((np.log(1 - i) + 7) * 37)))

    plt.xlim(dt_range)
    plt.ylim(total_time_range)
    plt.xlabel('time interval (s)')
    plt.ylabel('total time (s)')
    plt.title(f'isoprobability P(in focus), D: {diff_c:.3g} μm$^2$/s')
    plt.legend()


def plot_time_diff(dt: float, r: float, dr: float, diff_c_range: Sequence[float, float]) -> None:
    cmap = ListedColormap(colorcet.diverging_gkr_60_10_c40)
    diff_c = np.logspace(*(np.log10(diff_c_range)))
    cs = 0.999, 0.99, 0.95, 0.75, 0.5, 0.25
    for i in cs:
        plt.loglog(diff_c, total_time(diff_c, r, dr, dt, i), label=f"p: {i}",
                   color=cmap(int((np.log(1 - i) + 7) * 37)))

    plt.xlim(diff_c_range)
    plt.xlabel('D (μm$^2$/s)')
    plt.ylabel('total time (s)')
    plt.title('isoprobability P(in focus), interval: 5 s')
    plt.legend()
