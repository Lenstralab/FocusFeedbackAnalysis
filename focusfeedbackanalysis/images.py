from __future__ import annotations

import cv2
import numpy as np
import scipy.ndimage
from numpy.typing import ArrayLike


def get_nearest_px_msk(msk: ArrayLike, p: ArrayLike = None) -> tuple[np.ndarray, ...]:
    """Get the location of the nearest pixel in a mask msk to location p
    wp@tl20190927
    """
    p = p or np.array(msk.shape) / 2
    msk = msk.copy().astype(float)
    msk[msk > 0] = 1
    msk[msk == 0] = np.nan
    y, x = np.meshgrid(range(msk.shape[0]), range(msk.shape[1]))
    d = (x - p[0]) ** 2 + (y - p[1]) ** 2
    return np.unravel_index(np.nanargmin(d * msk), msk.shape)


def disk(s: int, dim: int = 2) -> np.ndarray:
    """make a disk shaped structural element to be used with
    morphological functions
    wp@tl20190709
    """
    d = np.zeros((s,) * dim)
    c = (s - 1) / 2
    mg = np.meshgrid(*(range(s),) * dim)
    d2 = np.sum([(i - c) ** 2 for i in mg], 0)
    d[d2 < s**2 / 4] = 1
    return d


def approxcontour(im: ArrayLike) -> np.ndarray:
    """usage: c = approxcontour(im)
    matlab: wp@tl20190522
    python: wp@tl20190710
    opencv: wp@tl20191101
    """
    im = np.asarray(im)
    lbl = set(im.flatten())
    lbl.remove(0)
    x = np.array(())
    y = np.array(())
    for l in lbl:
        d = im == l
        c, _ = cv2.findContours(d.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # type: ignore
        for i in range(len(c)):
            x = np.hstack((x, c[i][:, 0, 0], c[i][0, 0, 0], np.nan))
            y = np.hstack((y, c[i][:, 0, 1], c[i][0, 0, 1], np.nan))
    return np.vstack((x[:-1], y[:-1])).T


def gfilter(im: ArrayLike, sigma: float, r: float = 1.1) -> np.ndarray:
    """Bandpass filter an image using gaussian filters
    im:    2d array
    sigma: feature size to keep
    r:     lb, ub = sigma/r, sigma*r

    wp@tl2019
    """
    jm = np.array(im, copy=True)
    jm -= scipy.ndimage.gaussian_filter(jm, sigma * r)  # type: ignore
    return scipy.ndimage.gaussian_filter(jm, sigma / r)  # type: ignore


def crop(im: ArrayLike, x: ArrayLike, y: ArrayLike = None, z: ArrayLike = None, m: float = np.nan) -> np.ndarray:
    """crops image im, limits defined by min(x)..max(y), when these limits are
    outside im the resulting pixels will be filled with mean(im)
    wp@tl20181129
    """
    if isinstance(x, np.ndarray) and x.shape == (3, 2):
        z = x[2, :].copy().astype("int")
        y = x[1, :].copy().astype("int")
        x = x[0, :].copy().astype("int")
    elif isinstance(x, np.ndarray) and x.shape == (2, 2):
        y = x[1, :].copy().astype("int")
        x = x[0, :].copy().astype("int")
    else:
        x = np.array(x).astype("int")
        y = np.array(y).astype("int")
    if not z is None:  # 3D
        z = np.array(z).astype("int")
        s = np.array(np.shape(im))
        r0 = np.array([[min(y), max(y)], [min(x), max(x)], [min(z), max(z)]]).astype("int")
        r1 = r0.copy()
        r1[r0[:, 0] < 0, 0] = 1
        r1[r0[:, 1] > s, 1] = s[r0[:, 1] > s]
        jm = im[r1[0, 0] : r1[0, 1], r1[1, 0] : r1[1, 1], r1[2, 0] : r1[2, 1]]
        jm = np.concatenate(
            (
                np.full((r1[0, 0] - r0[0, 0], jm.shape[1], jm.shape[2]), m),
                jm,
                np.full((r0[0, 1] - r1[0, 1], jm.shape[1], jm.shape[2]), m),
            ),
            0,
        )
        jm = np.concatenate(
            (
                np.full((jm.shape[0], r1[1, 0] - r0[1, 0], jm.shape[2]), m),
                jm,
                np.full((jm.shape[0], r0[1, 1] - r1[1, 1], jm.shape[2]), m),
            ),
            1,
        )
        return np.concatenate(
            (
                np.full((jm.shape[0], jm.shape[1], r1[2, 0] - r0[2, 0]), m),
                jm,
                np.full((jm.shape[0], jm.shape[1], r0[2, 1] - r1[2, 1]), m),
            ),
            2,
        )
    else:  # 2D
        s = np.array(np.shape(im))
        r0 = np.array([[min(y), max(y)], [min(x), max(x)]]).astype(int)
        r1 = r0.copy()
        r1[r0[:, 0] < 1, 0] = 1
        r1[r0[:, 1] > s, 1] = s[r0[:, 1] > s]
        jm = im[r1[0, 0] : r1[0, 1], r1[1, 0] : r1[1, 1]]
        jm = np.concatenate(
            (
                np.full((r1[0, 0] - r0[0, 0], np.shape(jm)[1]), m),
                jm,
                np.full((r0[0, 1] - r1[0, 1], np.shape(jm)[1]), m),
            ),
            0,
        )
        return np.concatenate(
            (
                np.full((np.shape(jm)[0], r1[1, 0] - r0[1, 0]), m),
                jm,
                np.full((np.shape(jm)[0], r0[1, 1] - r1[1, 1]), m),
            ),
            1,
        )


def corrfft(im: ArrayLike, jm: ArrayLike) -> tuple[list[int], np.ndarray]:
    """usage: d, cfunc = corrfft(images)
    input:
        im, jm: images to be correlated
    output:
        d:      offset (x,y) in px
        cfunc:  correlation function
    """

    im -= np.nanmean(im)
    im /= np.nanstd(im)
    jm -= np.nanmean(jm)
    jm /= np.nanstd(jm)

    im, jm = im_max_size(im, jm)

    im[np.isnan(im)] = 0
    jm[np.isnan(jm)] = 0

    n_y = np.shape(im)[0]
    n_x = np.shape(im)[1]

    cfunc = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(im) * np.conj(np.fft.fft2(jm)))))
    y, x = np.unravel_index(np.nanargmax(cfunc), cfunc.shape)

    d = [x - np.floor(n_x / 2), y - np.floor(n_y / 2)]

    # peak at x=nX-1 means xoffset=-1
    if d[0] > n_x / 2:
        d[0] -= n_x
    if d[1] > n_y / 2:
        d[1] -= n_y

    return d, cfunc  # type: ignore


def im_max_size(*im):
    s = [jm.shape for jm in im]
    s = np.reshape(s, (len(s), len(s[0])))
    s = np.max(s, 0)
    jm = list()
    for i in im:
        p = ((0, s[0] - i.shape[0]), (0, s[1] - i.shape[1]))
        if np.all([j[1] == 0 for j in p]):
            jm.append(i)
        else:
            jm.append(np.pad(i, ((0, s[0] - i.shape[0]), (0, s[1] - i.shape[1])), "constant"))
    return jm
