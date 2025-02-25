from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike
from tqdm.auto import tqdm


def deltuple(t: tuple[Any, ...], n: int) -> tuple[Any, ...]:
    return t[:n] + t[n + 1 :]


def rmnan(*a: Any) -> tuple[Any, ...]:
    a = list(a)
    idx = np.full(0, 0)
    for i in range(len(a)):
        idx = np.append(idx, np.where(~np.isfinite(a[i])))
    idx = list(np.unique(idx))
    if len(idx):
        for i in range(len(a)):
            if hasattr(a[i], "__getitem__"):
                for j in reversed(idx):
                    if isinstance(a[i], tuple):
                        a[i] = deltuple(a[i], j)
                    elif isinstance(a[i], np.ndarray):
                        a[i] = np.delete(a[i], j)
                    else:
                        del a[i][j]
    return tuple(a)


def maskpk(pk: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """remove points in nx2 array which are located outside mask
    wp@tl20190709
    """
    pk = np.round(pk)
    idx = []
    for i in range(pk.shape[0]):
        if mask[tuple(pk[i, : mask.ndim])]:
            idx.append(i)
    return pk[idx, :]


def fixpar(n: int, fix: dict[int, float]) -> Callable[[ArrayLike], np.ndarray]:
    """Returns a function which will add fixed parameters in fix into an array
    N: total length of array which will be input in the function
    fix: dictionary, {2: 5.6}: fix parameter[2] = 5.6

    see its use in functions.fitgauss

    wp@tl20190816
    """
    # indices with variable parameters
    idx = sorted(list(set(range(n)) - set(fix)))

    # put the fixed paramters in place
    f = np.zeros(n)
    for i, v in fix.items():
        f[i] = v

    # make array used to construct variable part
    p = np.zeros((n, len(idx)))
    for i, j in enumerate(idx):
        p[j, i] = 1

    return lambda par: np.dot(p, par) + f


def unfixpar(p: Sequence[float], fix: dict[int, float]) -> np.ndarray:
    """reverse of fixpar, but just returns the array immediately instead of returning
    a function which will do it

    wp@tl20190816
    """
    p = list(p)
    [p.pop(i) for i in sorted(list(fix), reverse=True)]
    return np.array(p)


def errwrap(fun: Callable, default: Any = None, *args: Any, **kwargs: Any) -> Any:
    """Run a function fun, and when an error is caught return the default value
    wp@tl20190321
    """
    try:
        return fun(*args, **kwargs)
    except Exception:  # noqa
        return default


class TqdmMeter(tqdm):
    """Overload tqdm to make a special version of tqdm functioning as a meter."""

    def __init__(
        self, iterable: Iterable = None, desc: str = None, total: int = None, *args: Any, **kwargs: Any
    ) -> None:
        self._n = 0
        self._total = total
        self.disable = False
        if "bar_format" not in kwargs and len(args) < 16:
            kwargs["bar_format"] = "{desc}{bar}{n}/{total}"
        super().__init__(iterable, desc, total, *args, **kwargs)

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, value: int) -> None:
        if not value == self.n:
            self._n = int(value)
            self.refresh()

    @property
    def total(self) -> int:
        return self._total

    @total.setter
    def total(self, value: int) -> None:
        self._total = value
        if hasattr(self, "container"):
            self.container.children[1].max = value

    def __exit__(self, *args: Any, **kwargs: Any):
        if not self.leave:
            self.n = self.total
        super().__exit__(*args, **kwargs)
