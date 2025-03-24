from __future__ import annotations

import os
import pickle
from itertools import product
from pathlib import Path
from pprint import pprint
from typing import Any, Sequence, SupportsIndex

import matplotlib.pyplot as plt
import numpy as np
import pandas
import skimage
import yaml
from focusfeedbackgui.cylinderlens import find_z
from matplotlib.backends.backend_pdf import PdfPages
from ndbioimage import Imread
from numpy.typing import ArrayLike
from parfor import parfor, pmap
from scipy import ndimage, optimize, special
from tllab_common.fit import fminerr
from tqdm.auto import tqdm

from . import functions, images
from . import nbhelpers as nb
from . import utilities

if hasattr(yaml, "full_load"):
    yamlload = yaml.full_load
else:
    yamlload = yaml.load


def plot_localisations(f: pandas.DataFrame, im: Imread, c: int, z: int, t: int, pd: bool = True) -> None:
    im = im(c, z, t)
    plt.imshow(im, cmap="gray")
    h = f.query(f"C=={c} & Z=={z} & T=={t}")
    plt.plot(h["x"], h["y"], "or", markersize=15, markerfacecolor="None")
    plt.text(10, 25, str(t), color="white", size=25)
    plt.axis("off")
    plt.box(False)
    plt.gca().set(frame_on=False)
    plt.tick_params(
        axis="both",
        left=False,
        top=False,
        right=False,
        bottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
        labelbottom=False,
    )
    if pd:
        pprint(h)


def attach_units(f: pandas.DataFrame, im: Imread | float) -> pandas.DataFrame:
    """Calculate units for some columns in a localisation dataframe
    f:  dataframe containing localisations
    im: Imread object from which the localisations stem, or pxsize in um
    """
    f = f.copy()

    if hasattr(im, "pxsize_um"):
        pxsize = im.pxsize_um
    else:
        pxsize = im

    um = ["x", "y", "s_ini", "x_ini", "y_ini", "s", "r"]
    for i in um:
        if i in f:
            f[f"{i}_um"] = pxsize * f[i]
        if f"d{i}" in f:
            f[f"d{i}_um"] = pxsize * f[f"d{i}"]
        # if '{}_a'.format(i) in f:
        #    f['{}_a_um'.format(i)] = pxsize*f['{}_a'.format(i)]
        # if 'd{}_a'.format(i) in f:
        #    f['d{}_a_um'.format(i)] = pxsize*f['d{}_a'.format(i)]
    umi = ["tiltx", "tilty"]
    for i in umi:
        if i in f:
            f[f"{i}_um-1"] = 1 / pxsize * f[i]
        if f"d{i}" in f:
            f[f"d{i}_um-1"] = 1 / pxsize * f[f"d{i}"]
        # if '{}_a'.format(i) in f:
        #    f['{}_a_um-1'.format(i)] = 1/pxsize*f['{}_a'.format(i)]
        # if 'd{}_a'.format(i) in f:
        #    f['d{}_a_um-1'.format(i)] = 1/pxsize*f['d{}_a'.format(i)]
    return f


def insert_z(
    f: pandas.DataFrame,
    im: Imread,
    channels: Sequence[int],
    piezoval: pandas.DataFrame,
    timeval: Sequence[float],
    q: Sequence[float] = None,
    dq: Sequence[float] = None,
) -> pandas.DataFrame:
    """Calculate z from ellipticity and insert into the dataframe
    Also insert the time of recording of each frame

    wp@tl20190826
    """
    f = f.copy()
    pz = piezoval
    if pz.empty:
        pz.loc[0] = (0, 0, 0)
    if q is None:
        if hasattr(im.extrametadata, "q"):
            q = im.extrametadata.q
        else:
            q = [
                1.11350728,
                -0.11643823,
                0.15658226,
                1.26478436,
                0.48382135,
                0.41210216,
                3.73683468,
                21.41761362,
                0.97043002,
            ]
    if dq is None:
        dq = [i / 100 for i in q]
    f = f.copy()
    f.loc[:, "z_piezo"] = 0
    f.loc[:, "z_stage"] = 0
    f.loc[:, "z_ell"] = 0
    f.loc[:, "dz_ell"] = 0

    # first make sure we know the values for each frame
    for frame in f["T"].values:
        if frame not in pz["frame"].values:
            if frame - 1 in pz["frame"].values:
                pz = pandas.concat(
                    (
                        pz,
                        pandas.DataFrame(
                            (
                                (
                                    frame,
                                    float(pz[pz["frame"] == frame - 1].iloc[0]["piezoZ"]),
                                    float(pz[pz["frame"] == frame - 1].iloc[0]["stageZ"]),
                                ),
                            ),
                            columns=["frame", "piezoZ", "stageZ"],
                        ),
                    ),
                    ignore_index=True,
                )
            else:
                fframe = np.min(pz["frame"].values)
                pz = pandas.concat(
                    (
                        pz,
                        pandas.DataFrame(
                            (
                                (
                                    frame,
                                    float(pz[pz["frame"] == fframe].iloc[0]["piezoZ"]),
                                    float(pz[pz["frame"] == fframe].iloc[0]["stageZ"]),
                                ),
                            ),
                            columns=["frame", "piezoZ", "stageZ"],
                        ),
                    ),
                    ignore_index=True,
                )

    frame = f["T"].astype(int).tolist()
    ell = f["e"].tolist()
    dell = f["de"].tolist()
    channel = f["C"].astype("int").tolist()

    n = np.inf  # TODO: fix parallel case
    if len(ell) > n:  # calculate in parallel

        @parfor(
            zip(ell, dell, channel, frame),
            (q, dq, channels, pz, timeval),
            length=len(channel),
            desc="Calculating z from ell",
        )
        def qs(i, q, dq, channels, pz, t):  # noqa
            ell, dell, C, frame = i  # noqa
            z_piezo = [float(pz[pz["frame"] == fr].iloc[0]["piezoZ"]) for fr in frame]
            z_stage = [float(pz[pz["frame"] == fr].iloc[0]["stageZ"]) for fr in frame]
            t1 = [t[fr] - t[0] for fr in frame]
            e, de = zip(*[find_z(e, q, de, dq) if c in channels else (np.nan,) * 2 for e, de, c in zip(ell, dell, C)])
            return z_piezo, z_stage, t1, e, de

        rs = [[], [], [], [], []]
        for q in qs:
            for i, r in enumerate(q):
                rs[i].extend(r)
        f["z_piezo"], f["z_stage"], f["t"], f["z_ell"], f["z_dell"] = rs
    else:  # just calculate serially
        f["z_piezo"] = [float(pz[pz["frame"] == fr].iloc[0]["piezoZ"]) for fr in frame]
        f["z_stage"] = [float(pz[pz["frame"] == fr].iloc[0]["stageZ"]) for fr in frame]
        f["t"] = [timeval[fr] - timeval[0] for fr in frame]
        f["z_ell"], f["z_dell"] = zip(
            *[find_z(e, q, de, dq) if c in channels else (np.nan,) * 2 for e, de, c in zip(ell, dell, channel)]
        )

    f["z_um"] = f["z_stage"] + f["z_piezo"] + f["z_ell"]
    f["dz_um"] = f["dz_ell"]
    return f


def gaussian_1d(p: Sequence[float], x: float | np.ndarray) -> np.ndarray:
    """p: (mu, sigma, A, O)"""
    if len(p) == 3:
        return p[2] / p[1] / np.sqrt(2 * np.pi) * np.exp(-((x - p[0]) ** 2) / 2 / p[1] ** 2)
    return p[2] / p[1] / np.sqrt(2 * np.pi) * np.exp(-((x - p[0]) ** 2) / 2 / p[1] ** 2) + p[3]


def calibrate_intensity(
    im: Imread,
    files: Sequence[str | Path],
    calibration_path: str | Path,
    channels: Sequence[int],
    piezoval: pandas.DataFrame,
    timeval: Sequence[float],
    replace: bool = False,
) -> tuple[float, float, float, float]:
    """usage: z0, s, dz0, ds = calib_I(im)
    im: Imread instance or path to image file
    z0, s, dz0, ds: calibration parameters and their sER
    """
    calibration_path.mkdir(parents=True, exist_ok=True)
    pdfpath = calibration_path / "intensity_calib.pdf"
    ymlpath = calibration_path / "intensity_calib.yml"

    if not replace and ymlpath.exists():
        with open(ymlpath) as f:
            r = yamlload(f)
        return r["z0"], r["s"], r["dz0"], r["ds"]

    # Find bead files and load localisations
    ress = []
    for file in files:
        with open(calibration_path / file.with_suffix(".cyllens_calib.pk").name, "rb") as stream:
            ress.append(pickle.load(stream))

    # Combine results from bead files
    p = 0
    a = []
    for res in ress:
        if len(res):
            res["detections"]["particle"] += p
            p = res["detections"]["particle"].max() + 1
            a.append(res["detections"])
    a = pandas.concat(a)
    b = insert_z(a, im, channels, piezoval, timeval)
    b["z_piezo"] = a["z_um"]
    a = b

    @parfor(a["particle"].unique())
    def qs(pr: int) -> tuple[float, ...]:  # noqa
        a0 = a.query(f"particle=={pr} & R2>0").copy()  # noqa
        if len(a0) == 0:
            return pr, np.nan, np.nan, np.nan, np.nan, np.nan
        z = a0["z_piezo"]  # noqa
        i = a0["i_peak"]  # noqa

        q = [np.sum(z * i) / np.sum(i)]  # noqa
        q.append(np.sqrt(np.sum(i**2 * (z - q[0]) ** 2) / np.sum(i**2)))
        q.append(np.trapz(i, z))
        q.append(0)
        g = lambda p: np.nansum((i - gaussian_1d(p, z)) ** 2)  # noqa
        r = optimize.minimize(g, np.array(q), options={"disp": False, "maxiter": 1e5})  # noqa
        p = r.x  # noqa
        R2 = 1 - np.nansum((i - gaussian_1d(p, z)) ** 2) / np.sum((i - np.mean(i)) ** 2)  # noqa
        return pr, *p, R2

    qs = np.array(qs)
    f = pandas.DataFrame(
        {
            "particle": qs[:, 0].astype("int"),
            "mu": qs[:, 1],
            "sigma": qs[:, 2],
            "intensity": qs[:, 3],
            "offset": qs[:, 4],
            "R2": qs[:, 5],
        }
    )

    # plt.hist(f['R2'], 100, range=(0,1));
    h = f.query("R2>0.8").copy()

    intensity = []
    zs = []
    z2s = []
    ellipticity = []
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2)
    fig.add_subplot(gs[0, :])

    for pr in tqdm(h["particle"].unique()):
        a0 = a.query(f"particle=={pr}").copy()
        g0 = h.query(f"particle=={pr}").copy()
        z = np.array(a0["z_piezo"]) - float(g0["mu"])
        i = (a0["i_peak"] - float(g0["offset"])) * np.sqrt(2 * np.pi) * float(g0["sigma"]) / float(g0["intensity"])
        plt.plot(z, i, ".")
        intensity.extend(i)
        zs.extend(z)
        z2s.extend(a0["z_ell"])
        ellipticity.extend(a0["e"])

    intensity = np.array(intensity).flatten()
    zs = np.array(zs).flatten()
    z2s = np.array(z2s).flatten()
    ellipticity = np.array(ellipticity).flatten()

    # print('len h: {}'.format(len(h)))
    # print('len Z: {}'.format(len(Z)))
    # plt.figure()
    # plt.plot(Z, Z2);
    # plt.xlabel('gfvb dg')
    # plt.figure()

    ellipticity = ellipticity[np.abs(zs) < 0.3]
    z2s = z2s[np.abs(zs) < 0.3]
    intensity = intensity[np.abs(zs) < 0.3]
    zs = zs[np.abs(zs) < 0.3]

    z0, intensity0 = utilities.rmnan(zs, intensity)

    q = [np.sum(z0 * intensity0) / np.sum(intensity0)]  # noqa
    q.append(np.sqrt(np.sum(intensity0**2 * (z0 - q[0]) ** 2) / np.sum(intensity0**2)))
    q.append(np.trapz(intensity0, z0) / (2 * np.pi * 5))
    g = lambda p: np.nansum((intensity0 - gaussian_1d(p, z0)) ** 2)  # noqa
    r = optimize.minimize(g, np.array(q), options={"disp": False, "maxiter": 1e5})
    p = r.x
    dp, r2 = fminerr(lambda p: gaussian_1d(p, z0), p, intensity0)  # noqa

    zz = np.linspace(-1, 1, 250)
    plt.plot(zz, gaussian_1d(p, zz.astype(float)), "-r")
    plt.xlim(-1, 1)
    plt.xlabel(r"z (μm)")
    plt.ylabel("peak intensity")

    # Z, Z2, I, E = utilities.rmnan(Z, Z2, I, E)
    z0 = zs.copy()
    z0[z0 == 0] = np.nan
    z20 = z2s.copy()

    d, t, g2, dd, dt = functions.linefitdt(*utilities.rmnan(z0, z20))
    m = (np.tan(t), -d / np.cos(t))
    dm = (np.abs(dt / np.cos(t) ** 2), np.sqrt(d**2 * np.tan(t) ** 2 * dt**2 + dd**2) / np.abs(np.cos(t)))

    x = np.linspace(-0.3, 0.3)
    fig.add_subplot(gs[1, 0])
    plt.plot(zs, z2s, ".")
    plt.plot(x, np.polyval(m, x), "-r")
    plt.plot()
    plt.xlabel(r"z (μm)")
    plt.ylabel(r"z_ell (μm)")
    print("Linefit: z_ell = {}z + {}".format(*m))

    if ellipticity.size:
        fit = np.polyfit(*utilities.rmnan(ellipticity, intensity), 2)
    else:
        fit = np.full(3, np.nan)
    x = np.linspace(0.5, 1.5)
    fig.add_subplot(gs[1, 1])
    plt.plot(ellipticity, intensity, ".")
    plt.plot(x, np.polyval(fit, x), "-r")
    plt.xlabel("ellipticity")
    plt.ylabel("peak intensity")

    print(f"p0: {p[0]}\nm1: {m[1]}")

    pe = (m[0] * p[0] + m[1], m[0] * p[1], m[0] * p[1] * np.sqrt(2 * np.pi))
    dpe = (
        np.sqrt(m[0] ** 2 * dp[0] ** 2 + dm[0] ** 2 * p[0] ** 2 + dm[1] ** 2),
        np.sqrt(m[0] ** 2 * dp[1] ** 2 + dm[0] ** 2 * p[1] ** 2),
        np.sqrt(m[0] ** 2 * dp[1] ** 2 + dm[0] ** 2 * p[1] ** 2) * np.sqrt(2 * np.pi),
    )

    zz = np.linspace(np.nanmin(z2s), np.nanmax(z2s), 250)
    fig.add_subplot(gs[2, :])
    plt.plot(z2s, intensity, ".")
    fit = plt.plot(zz, gaussian_1d(pe, zz.astype(float)), "-r")
    plt.xlabel(r"z_ell (μm)")
    plt.ylabel("peak intensity")
    plt.legend(fit, ("z0: {:.2f} um\ns:  {:.2f} um\nA: {:.2f}".format(*pe),))

    print(f"z0: {pe[0]:.2f} +- {dpe[0]:.2f}")
    print(f"s:  {pe[1]:.2f} +- {dpe[1]:.2f}")

    with PdfPages(pdfpath) as pdf:
        pdf.savefig(fig)

    pe = [float(i) for i in pe]
    dpe = [float(i) for i in dpe]
    with open(ymlpath, "w") as f:
        yaml.safe_dump({"z0": pe[0], "dz0": dpe[0], "s": pe[1], "ds": dpe[1]}, f, default_flow_style=None)

    return pe[0], pe[1], dpe[0], dpe[1]


def correct_intensity(*args: Any) -> pandas.DataFrame | tuple[np.ndarray, np.ndarray]:
    """usage: I, dI = correct_I(I, Z, z0, s, dI, dZ, dz0, ds, maxdz)
           f = correct_I(f, z0, s, dz0, ds)
    Calculates corrected peak intensity for out of focus localisations (position Z, Z=z0: in focus)
        based on a Gaussian fit with center z0 and width (sigma) s.
        Make sure that any dataframe only contains the channel that needs to be corrected!
    I, Z, dI, dZ: arrays with peak intensity, z calculated from ellipticity and their sER
    f: dataframe with columns i_peak, z_ell, di_peak, dz_ell
    Also propagates errors
    20200611
    """

    def parse(intensity, z, z0, s, d_intensity=np.nan, dz=np.nan, dz0=np.nan, ds=np.nan, maxdz=np.inf):  # noqa
        return intensity, z, z0, s, d_intensity, dz, dz0, ds, maxdz

    def parse2(f, z0, s, dz0, ds, maxdz=np.inf):  # noqa
        return f, z0, s, dz0, ds, maxdz

    if isinstance(args[0], (pandas.DataFrame, pandas.Series)):
        f, z0, s, dz0, ds, maxdz = parse2(*args)
        i_peak = f["i_peak"].to_numpy()
        z_ell = f["z_ell"].to_numpy()
        di_peak = f["di_peak"].to_numpy()
        dz_ell = f["dz_ell"].to_numpy()
    else:
        f = None
        i_peak, z_ell, z0, s, di_peak, dz_ell, dz0, ds, maxdz = parse(*args)

    dz = np.clip(z_ell - z0, -maxdz, maxdz)

    ellipticity = np.exp(dz**2 / 2 / s**2)
    di_peak = ellipticity * np.sqrt(
        di_peak**2 + i_peak**2 * dz**2 / s**4 * (dz_ell**2 + dz0**2) + i_peak**2 * dz**4 / s**6 * ds**2
    )
    i_peak = ellipticity * i_peak

    if f is not None:
        f["i_peak_uc"], f["di_peak_uc"] = f["i_peak"], f["di_peak"]
        f["i_peak"], f["di_peak"] = i_peak, di_peak
        return f
    else:
        return i_peak, di_peak


def merge_dataframes(a: pandas.DataFrame, b: pandas.DataFrame, identifier: str = "_a") -> pandas.DataFrame:
    """Merge two dataframes containing localisation data
    a, b:       dataframes
    identifier: some columns are copied from a to b and identifier
                  is appended to their name

    wp@tl20190823
    """
    b = b.copy()
    a = a.loc[b.index]
    transfer = ["i", "o", "tiltx", "tilty", "i_peak", "X2", "R2", "sn"]
    for i in transfer:
        if i in a:
            b[f"{i}{identifier}"] = a[i]
        if f"d{i}" in a:
            b[f"d{i}{identifier}"] = a[f"d{i}"]

    replace = ["x", "y", "s", "e", "theta", "x_ini", "y_ini"]
    for i in replace:
        if i in a:
            b[i] = a[i]
        if f"d{i}" in a:
            b[f"d{i}"] = a[f"d{i}"]
    return b


def detect_points_sf(
    im: ArrayLike, sigma: float = None, mask: ArrayLike = None, footprint: int = 15, filter_peaks: bool = True
) -> pandas.DataFrame:
    """Find interesting spots to which try to fit gaussians in a single frame
    im:      2D or 3D array
    sigma:   theoretical width of the psf (isotropic)
    mask:    logical 2D or 3D array masking the area in which points are to be found

    wp@tl201908
    """
    dim = np.ndim(im)

    # pk = skimage.feature.blob_log(im, 1, 3, 21)
    # pk = pk[1<pk[:,2]]
    # pk = pk[pk[:,2]<3]

    if sigma is None:
        c = im
    else:
        # c = images.gfilter(im, sigma)
        c = ndimage.gaussian_laplace(im, sigma)
        c = ndimage.gaussian_filter(-c, sigma)

    pk = skimage.feature.peak_local_max(c, footprint=images.disk(footprint, dim))
    if mask is not None:
        pk = utilities.maskpk(pk, mask)

    # plt.imshow(c)
    f = pandas.DataFrame({"y_ini": pk[:, 0], "x_ini": pk[:, 1]})
    if dim == 3:
        f["z_ini"] = pk[:, 2]

    # plt.plot(f['x_ini'], f['y_ini'], 'ro', markersize=15, markerfacecolor='None')

    p = []
    r2 = []
    for i in range(len(f)):
        g = f.loc[i]
        if dim == 3:
            jm = images.crop(im, g["x_ini"] + [-8, 8], g["y_ini"] + [-8, 8], g["z_ini"] + [-8, 8])
        else:
            jm = images.crop(im, g["x_ini"] + [-8, 8], g["y_ini"] + [-8, 8])
        p.append(fitgaussint(jm))
        if dim == 3:
            r2.append(1 - np.nansum((gaussian(p[-1], 16, 16, 16) - jm) ** 2) / np.nansum((jm - np.nanmean(jm)) ** 2))
        else:
            r2.append(1 - np.nansum((gaussian(p[-1], 16, 16) - jm) ** 2) / np.nansum((jm - np.nanmean(jm)) ** 2))
    p = np.array(p)
    if len(f):
        f["y"] = p[:, 0] + f["y_ini"] - 8
        f["x"] = p[:, 1] + f["x_ini"] - 8
        f["R2"] = r2
        if dim == 2:
            f["s_ini"] = p[:, 2] / 2 / np.sqrt(2 * np.log(2))
            f["i_ini"] = p[:, 3]
            f["o_ini"] = p[:, 4]
            f["e_ini"] = p[:, 5]
            f["theta_ini"] = p[:, 6]
        else:
            f["z"] = p[:, 2] + f["z_ini"] - 8
            f["s_ini"] = p[:, 3] / 2 / np.sqrt(2 * np.log(2))
            f["sz_ini"] = p[:, 4] / 2 / np.sqrt(2 * np.log(2))
            f["i_ini"] = p[:, 5]
            f["o_ini"] = p[:, 6]
    else:
        f["y"] = []
        f["x"] = []
        if dim == 2:
            f["e_ini"] = []
            f["theta_ini"] = []
        else:
            f["z"] = []
            f["sz_ini"] = []
        f["s_ini"] = []
        f["i_ini"] = []
        f["o_ini"] = []
        f["R2"] = []

    if filter_peaks:
        f = f.dropna()
        if len(f) > 5:
            s = sorted(f["i_ini"] / f["s_ini"] ** 2)
            th = min(skimage.filters.threshold_otsu((f["i_ini"] / f["s_ini"] ** 2).to_numpy()), s[-5])
            f = f.query(f"i_ini/s_ini**2>{th}")
    # f = f.query('R2>0')
    return f


def detect_points(
    im: Imread | np.ndarray, sigma: Sequence[float], mask: np.ndarray = None, ndim: int = 2, **kwargs: Any
) -> pandas.DataFrame:
    """Iteratively finds spots in an nd-array im (numpy or Imread)
    mask: 2d or 3d array, or list or tuple of 2d or 3d arrays, one entry for each channel

    wp@tl201908
    """
    if isinstance(im, np.ndarray):
        im = Imread(im, axes="cztyx")

    def fun2(frame, im, mask):  # noqa
        s = sigma[frame[0]]
        if isinstance(mask, (list, tuple)):
            f = detect_points_sf(im(*frame), s, mask[frame[0]], **kwargs)
        else:
            f = detect_points_sf(im(*frame), s, mask, **kwargs)
        f["C"], f["Z"], f["T"] = frame
        return f

    def fun3(frame, im, mask):  # noqa
        s = sigma[frame[0]]
        if mask.zstack:
            mask = mask.transpose("ctyxz")  # noqa
        else:
            mask = mask.transpose("ctyx")  # noqa
        f = detect_points_sf(im.transpose("ctyxz")[frame[0], frame[1]], s, mask[frame[0], frame[1]], **kwargs)
        f["C"], f["T"] = frame
        return f

    if ndim == 2:
        q = pandas.concat(
            pmap(
                fun2,
                product(*[range(i) for i in im.shape["czt"]]),
                (im, mask),
                total=im.shape["c"] * im.shape["z"] * im.shape["t"],
                desc="Detecting points in frames",
            ),
            sort=True,
        )
    else:
        q = pandas.concat(
            pmap(
                fun3,
                product(*[range(i) for i in im.shape["ct"]]),
                (im, mask),
                total=im.shape["c"] * im.shape["t"],
                desc="Detecting points in frames",
            ),
            sort=True,
        )
    q.index = range(q.shape[0])
    return q


def tpsuperresseq(
    f: pandas.DataFrame,
    im: Imread,
    sigma: SupportsIndex,
    theta: float | bool = True,
    tilt: bool = True,
    keep: Sequence[str] = None,
    desc: str = None,
    bar: bool = True,
    ndim: int = 2,
) -> pandas.DataFrame:
    """fit localizations in f
    im: Imread with images
    sigma / sigma_z: list with value for each channel
    """
    if keep is None:
        keep = []
    desc = desc or "Fitting localisations"
    fix = {}
    if ndim == 2:
        if theta is True:
            ell = True
        elif theta is None or theta is False:
            ell = False
        else:
            ell = True
            fix[6] = theta
        if tilt is not True:
            fix[7] = 0
            fix[8] = 0
        k = ["x", "y", "s", "i", "o", "e", "theta", "tiltx", "tilty"]
    else:
        ell = False
        if tilt is not True:
            fix[7] = 0
            fix[8] = 0
            fix[9] = 0
        k = ["x", "y", "z", "s", "sz", "i", "o", "tiltx", "tilty", "tiltz"]

    def fun(c, im, sigma, keep, fix, ell, tilt):  # noqa
        f = c[1].copy()  # noqa
        if isinstance(im, Imread):
            if ndim == 2:
                im = im.transpose("cztyx")[int(f["C"]), int(f["Z"]), int(f["T"])]  # noqa
            else:
                im = im.transpose("ctyxz")[int(f["C"]), int(f["T"])]  # noqa
        fwhm = sigma[int(f["C"])] * 2 * np.sqrt(2 * np.log(2))
        for j in keep:
            if j in k:
                if j == "s":
                    fix[k.index(j)] = float(2 * np.sqrt(2 * np.log(2)) * f[j])
                    fwhm = fix[k.index(j)]
                else:
                    fix[k.index(j)] = float(f[j])

        if ndim == 2:
            q, dq, s = fitgauss(im, np.array(f[["x", "y"]]), ell, tilt, fwhm, fix, pl=False)  # noqa
        else:
            q, dq, s = fitgauss(im, np.array(f[["x", "y", "z"]]), False, tilt, fwhm, fix, pl=False)  # noqa

        f["y_ini"] = f["y"]
        f["x_ini"] = f["x"]
        f["y"] = q[1]
        f["x"] = q[0]
        f["dy"] = dq[1]
        f["dx"] = dq[0]
        f["tiltx"] = q[7]
        f["dtiltx"] = dq[7]
        f["tilty"] = q[8]
        f["dtilty"] = dq[8]

        if ndim == 2:
            f["e"] = q[5]
            f["de"] = dq[5]
            f["theta"] = q[6]
            f["dtheta"] = dq[6]
            f["s"] = q[2] / (2 * np.sqrt(2 * np.log(2)))
            f["ds"] = dq[2] / (2 * np.sqrt(2 * np.log(2)))
            f["i"] = q[3]
            f["di"] = dq[3]
            f["o"] = q[4]
            f["do"] = dq[4]
        else:
            f["z_ini"] = f["z"]
            f["z"] = q[2]
            f["s"] = q[3] / (2 * np.sqrt(2 * np.log(2)))
            f["ds"] = dq[3] / (2 * np.sqrt(2 * np.log(2)))
            f["sz"] = q[4] / (2 * np.sqrt(2 * np.log(2)))
            f["dsz"] = dq[4] / (2 * np.sqrt(2 * np.log(2)))
            f["i"] = q[5]
            f["di"] = dq[5]
            f["o"] = q[6]
            f["do"] = dq[6]
            f["tiltz"] = q[9]
            f["dtiltz"] = dq[9]

        f["i_peak"] = f["i"] / (2 * np.pi * f["s"] ** 2)
        f["di_peak"] = f["i_peak"] * np.sqrt(4 * (f["ds"] / f["s"]) ** 2 + (f["di"] / f["i"]) ** 2)
        f["X2"] = s[0]
        f["R2"] = s[1]
        f["sn"] = s[2]

        return pandas.DataFrame(f).transpose()

    if isinstance(im, (tuple, list)):
        im, sigma = im
        assert len(f) == 1, "f should have only one row"
        assert im.ndim == ndim, f"im should have {ndim} dimensions"
        return fun((0, f.iloc[0]), np.asarray(im), sigma, keep, fix, ell, tilt)
    else:
        if len(f):
            q = pmap(fun, f.iterrows(), (im, sigma, keep, fix, ell, tilt), desc=desc, total=len(f), bar=bar)
            return pandas.concat(q, sort=True)
        else:
            return pandas.DataFrame(
                columns=list(
                    set(f.columns).union(
                        {
                            "y_ini",
                            "x_ini",
                            "dy",
                            "dx",
                            "s",
                            "ds",
                            "i",
                            "di",
                            "o",
                            "do",
                            "e",
                            "de",
                            "theta",
                            "dtheta",
                            "tiltx",
                            "dtiltx",
                            "tilty",
                            "dtilty",
                            "i_peak",
                            "X2",
                            "R2",
                            "sn",
                        }
                    )
                )
            )


def gaussian3d(p: Sequence[float], size_x: int, size_y: int, size_z: int) -> np.ndarray:
    """p: [x,y,z,fwhm,fwhmz,area,offset,x-tilt,y-tilt,z-tilt]
    X,Y,Z: size of image
    """
    dx = np.sqrt(np.log(2)) / p[3]
    dy = dx
    dz = np.sqrt(np.log(2)) / p[4]
    xv, yv, zv = np.meshgrid(np.arange(size_y) - p[0], np.arange(size_x) - p[1], np.arange(size_z) - p[2])

    x = 2 * dx * xv
    y = 2 * dy * yv
    z = 2 * dz * zv

    erf = special.erf
    if np.size(p) < 8:
        offset = p[6]
    else:
        offset = p[6] + xv * p[7] + yv * p[8] + zv * p[9]
    return p[5] / 8 * (erf(x + dx) - erf(x - dx)) * (erf(y + dy) - erf(y - dy)) * (erf(z + dz) - erf(z - dz)) + offset


def gaussian(p: Sequence[float], size_x: int, size_y: int, size_z: int = None) -> np.ndarray:  # noqa
    """p: [x,y,fwhm,area,offset,ellipticity,angle towards x-axis]
    default ellipticity & angle: 1 resp. 0
    X,Y: size of image
    reimplemented for numba, small deviations from true result
        possible because of reimplementation of erf for numba
    """
    p = tuple(p)
    if len(p) == 5:
        return nb.gaussian5(p, size_x, size_y)
    elif len(p) == 6:
        return nb.gaussian6(p, size_x, size_y)
    elif len(p) == 7:
        return nb.gaussian7(p, size_x, size_y)
    elif len(p) == 9:
        return nb.gaussian9(p, size_x, size_y)
    else:
        raise ValueError("not enough or to many parameters")


def fitgauss(
    im: np.ndarray,
    xy: Sequence[int] = None,
    ell: bool = False,
    tilt: bool = False,
    fwhm: float = None,
    fix: dict[int, float] = None,
    pl: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Fit gaussian function to image
    im:    2D array with image
    xy:    Initial guess for x, y, optional, default: pos of max in im
    ell:   Fit with ellipicity if True
    fwhm:  fwhm of the peak, used for boundary conditions
    fix:   dictionary describing which parameter to fix, to fix theta: fix={6: theta}
    q:  [x,y,fwhm,area,offset,ellipticity,angle towards x-axis,tilt-x,tilt-y]
    dq: errors (std) on q

    wp@tl2019
    """

    # print('xy:   ', xy)
    # print('ell:  ', ell)
    # print('tilt: ', tilt)
    # print('fwhm: ', fwhm)
    # print('fix:  ', fix)

    if fwhm is not None:
        fwhm = np.round(fwhm, 2)

    # handle input options
    if xy is None:
        # filter to throw away any background and approximate position of peak
        fm = (im - ndimage.gaussian_filter(im, 0.2))[1:-1, 1:-1]
        xy = [i + 1 for i in np.unravel_index(np.nanargmax(fm.T), np.shape(fm))]
    else:
        xy = [int(np.round(i)) for i in xy]
    if fix is None:
        fix = {}
    if ell is False:
        if 5 not in fix:
            fix[5] = 1
        if 6 not in fix:
            fix[6] = 0
    if tilt is False:
        if 7 not in fix:
            fix[7] = 0
        if 8 not in fix:
            fix[8] = 0

    xy = np.array(xy)
    for i in range(2):
        if i in fix:
            xy[i] = int(np.round(fix[i]))

    # size initial crop around peak
    if fwhm is None or not np.isfinite(fwhm):
        r = 10
    else:
        r = 2.5 * fwhm

    # find tilt parameters from area around initial crop
    if tilt:
        cc = np.round(((xy[0] - 2 * r, xy[0] + 2 * r + 1), (xy[1] - 2 * r, xy[1] + 2 * r + 1))).astype("int")
        km = images.crop(im, cc)
        k = [i / 2 for i in km.shape]
        km[
            int(np.ceil(k[0] - r)) : int(np.floor(k[0] + r + 1)), int(np.ceil(k[1] - r)) : int(np.floor(k[1] + r + 1))
        ] = np.nan
        t = fit_tilted_plane(km)
    else:
        t = [0, 0, 0]

    # find other initial parameters from initial crop with tilt subtracted
    cc = np.round(((xy[0] - r, xy[0] + r + 1), (xy[1] - r, xy[1] + r + 1))).astype("int")
    jm = images.crop(im, cc)
    xv, yv = nb.meshgrid2(*map(np.arange, jm.shape[::-1]))

    if 6 in fix:
        p = fitgaussint(jm - t[0] - t[1] * xv - t[2] * yv, theta=fix[6])
    else:
        p = fitgaussint(jm - t[0] - t[1] * xv - t[2] * yv)
    p[0:2] += cc[:, 0] + 1
    if pl:
        print("initial q: ", p)

    for i in range(2):
        if i in fix:
            p[i] = xy[i]

    if fwhm is None:
        fwhm = p[2]
    else:
        p[2] = fwhm

    # just give up in some cases
    if not 1 < p[2] < 2 * fwhm or p[3] < 0.1:
        q = np.full(9, np.nan)
        dq = np.full(9, np.nan)
        return q, dq, (np.nan, np.nan, np.nan)

    s = fwhm / np.sqrt(2)  # new crop size

    cc = np.round(((p[0] - s, p[0] + s + 1), (p[1] - s, p[1] + s + 1))).astype("int")
    jm = images.crop(im, cc)
    shape = np.shape(jm)

    bnds = [
        (0, shape[0] - 1),
        (0, shape[1] - 1),
        (fwhm / 2, fwhm * 2),
        (1e2, None),
        (0, None),
        (0.5, 2),
        (None, None),
        (None, None),
        (None, None),
    ]
    xv, yv = nb.meshgrid2(*map(np.arange, shape[::-1]))

    # move fixed x and/or y with the crop
    for i in range(2):
        if i in fix:
            fix[i] -= cc[i, 0]
            xy[i] = p[i]

    # find tilt from area around new crop
    cd = np.round(((p[0] - 2 * s, p[0] + 2 * s + 1), (p[1] - 2 * s, p[1] + 2 * s + 1))).astype("int")
    km = images.crop(im, cd)
    k = [i / 2 for i in km.shape]
    km[int(np.ceil(k[0] - s)) : int(np.floor(k[0] + s + 1)), int(np.ceil(k[1] - s)) : int(np.floor(k[1] + s + 1))] = (
        np.nan
    )
    t = fit_tilted_plane(km)

    # update parameters to new crop
    p[0:2] -= cc[:, 0]
    p = np.append(p, (t[1], t[2]))
    # p = np.append(p, (1, 0, t[1], t[2]))
    p[4] = t[0] + t[1] * (p[0] + s) + t[2] * (p[1] + s)

    # remove fixed parameters and bounds from lists of initial parameters and bounds
    p = utilities.unfixpar(p, fix)
    [bnds.pop(i) for i in sorted(list(fix), reverse=True)]

    # define function to remove fixed parameters from list, then define function to be minimized
    fp = utilities.fixpar(9, fix)
    g = lambda a: np.nansum((jm - nb.gaussian9grid(fp(a), xv, yv)) ** 2)  # noqa

    # make sure the initial parameters are within bounds
    for i, b in zip(p, bnds):
        utilities.errwrap(np.clip, i, i, b[0], b[1])

    n_par = len(p)

    # fit and find error predictions
    r = optimize.minimize(g, p, options={"disp": False, "maxiter": 1e5})

    q = r.x
    # dq = np.sqrt(r.fun/(np.size(jm)-np.size(q))*np.diag(r.hess_inv))
    if pl:
        q0 = fp(q)
        print("q after first fit: ", q0[:2] + cc[:, 0], q0[2:])
        print("nfev:", r.nfev)

    # Check boundary conditions, maybe try to fit again
    refitted = False
    for idx, (i, b) in enumerate(zip(q, bnds)):
        try:
            if not b[0] < i < b[1] and not refitted:
                r = optimize.minimize(g, p, options={"disp": False, "maxiter": 1e7}, bounds=bnds)
                q = r.x
                # dq = functions.fminerr(lambda p: nb.gaussian9grid(fp(p), xv, yv), q, jm)[1]
                if pl:
                    print("bounds {}: {} < {} < {}\nq after refit: {}\nnfev: {}".format(idx, b[0], i, b[1], q, r.nfev))
                    print("initial params: ", p)
                refitted = True
        except Exception:  # noqa
            pass

    if pl:
        print("Refitted: ", refitted)
        a, b = np.min(jm), np.max(jm)
        plt.figure()
        plt.imshow(jm, vmin=a, vmax=b)
        plt.plot(q0[0], q0[1], "or")  # noqa
        plt.figure()
        plt.imshow(nb.gaussian9grid(fp(q), xv, yv), vmin=a, vmax=b)
        plt.figure()
        plt.imshow(np.abs(nb.gaussian9grid(fp(q), xv, yv) - jm), vmin=0, vmax=b - a)

    # dq = functions.fminerr(lambda p: nb.gaussian9grid(fp(p), xv, yv), q, jm)[1]
    dq, _ = fminerr(lambda p: nb.gaussian9grid(p, xv, yv), fp(q), jm)  # noqa

    # reinsert fixed parameters
    q = fp(q)
    # for i in sorted(fix):
    #     if i > len(dq):
    #         dq = np.append(dq, 0)
    #     else:
    #         dq = np.insert(dq, i, 0)

    # de-degenerate parameters and recalculate position from crop to frame
    q[2] = np.abs(q[2])
    q[0:2] += cc[:, 0]
    q[5] = np.abs(q[5])
    # q[6] %= np.pi
    q[6] = (q[6] + np.pi / 2) % np.pi - np.pi / 2

    # Chi-squared, R-squared, signal to noise ratio
    chisq = r.fun / (shape[0] * shape[1] - n_par)
    r2 = 1 - r.fun / np.nansum((jm - np.nanmean(jm)) ** 2)
    sn = q[3] / np.sqrt(r.fun / (shape[0] * shape[1])) / 2 / np.pi / q[2] ** 2

    return q, dq, (float(chisq), float(r2), float(sn))


def fitgaussint(
    im: np.ndarray, xy: Sequence[int] = None, theta: float = None, mesh: tuple[np.ndarray, ...] = None
) -> np.ndarray:
    """finds initial parameters for a 2d Gaussian fit
    q = (x, y, fwhm, area, offset, ellipticity, angle) if 2D
    q = (x, y, z, fwhm, fwhmz, area, offset) if 3D
    wp@tl20191010
    """

    dim = np.ndim(im)
    shape = np.shape(im)
    q = np.full(7, 0).astype("float")

    if dim == 2:
        if mesh is None:
            x, y = np.meshgrid(range(shape[1]), range(shape[0]))
        else:
            x, y = mesh

        if theta is None:
            tries = 10
            e = []
            t = np.delete(np.linspace(0, np.pi, tries + 1), tries)
            for th in t:
                e.append(fitgaussint(im, xy, th, (x, y))[5])
            q[6] = (fitcosint(2 * t, e)[2] / 2 + np.pi / 2) % np.pi - np.pi / 2
        else:
            q[6] = theta

        # q[4] = np.nanmin(im)
        jm = im.flatten()
        jm = jm[np.isfinite(jm)]
        q[4] = np.mean(np.percentile(jm, 0.25))
        q[3] = np.nansum(im - q[4])

        if xy is None:
            q[0] = np.nansum(x * (im - q[4])) / q[3]
            q[1] = np.nansum(y * (im - q[4])) / q[3]
        else:
            q[:2] = xy

        cos, sin = np.cos(q[6]), np.sin(q[6])
        x, y = cos * (x - q[0]) - (y - q[1]) * sin, cos * (y - q[1]) + (x - q[0]) * sin

        s2 = np.nansum((im - q[4]) ** 2)
        sx = np.sqrt(np.nansum((x * (im - q[4])) ** 2) / s2)
        sy = np.sqrt(np.nansum((y * (im - q[4])) ** 2) / s2)

        q[2] = np.sqrt(sx * sy) * 4 * np.sqrt(np.log(2))
        q[5] = np.sqrt(sx / sy)
    else:
        if mesh is None:
            x, y, z = np.meshgrid(range(shape[0]), range(shape[1]), range(shape[2]))
        else:
            x, y, z = mesh
        q[6] = np.nanmin(im)
        q[5] = np.nansum(im - q[6])

        if xy is None:
            q[0] = np.nansum(x * (im - q[6])) / q[5]
            q[1] = np.nansum(y * (im - q[6])) / q[5]
            q[2] = np.nansum(z * (im - q[6])) / q[5]
        else:
            q[:3] = xy

        x, y, z = x - q[0], y - q[1], z - q[2]

        s2 = np.nansum((im - q[6]) ** 2)
        sx = np.sqrt(np.nansum((x * (im - q[6])) ** 2) / s2)
        sy = np.sqrt(np.nansum((y * (im - q[6])) ** 2) / s2)
        sz = np.sqrt(np.nansum((z * (im - q[6])) ** 2) / s2)

        q[3] = np.sqrt(sx * sy) * 4 * np.sqrt(np.log(2))
        q[4] = sz * 4 * np.sqrt(np.log(2))
    return q


def fitcosint(theta: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Finds parameters to y=a*cos(theta-psi)+b
    wp@tl20191010
    """
    b = np.trapz(y, theta) / np.mean(theta) / 2
    a = np.trapz(np.abs(y - b), theta) / 4

    t = np.sin(theta)
    s = np.cos(theta)

    ts = np.sum(t)
    ss = np.sum(s)
    as_ = np.sum(y * t)
    bs = np.sum(y * s)
    cs = np.sum(t**2)
    ds = np.sum(t * s)
    es = np.sum(s**2)

    q = np.dot(np.linalg.inv(((cs, ds), (ds, es))) / a, (as_ - b * ts, bs - b * ss))

    psi = (np.arctan2(*q)) % (2 * np.pi)
    if q[1] < 0:
        a *= -1
        psi -= np.pi
    psi = (psi + np.pi) % (2 * np.pi) - np.pi

    return np.array((a, b, psi))


def fit_tilted_plane(im: np.ndarray) -> np.ndarray:
    """Linear regression to determine z0, a, b in z = z0 + a*x + b*y

    nans and infs are filtered out

    input: 2d array containing z (x and y will be the pixel numbers)
    output: array [z0, a, b]

    wp@tl20190819
    """
    shape = im.shape
    im = im.flatten()
    if np.all(np.isnan(im)):
        raise ValueError

    # vector [1, x_ij, y_ij] and filter nans and infs
    mv = np.meshgrid(*map(range, shape))
    v = [i.flatten()[np.isfinite(im)] for i in [np.ones(im.shape), *mv]]
    # construct matrix for the regression
    q = np.array([[np.sum(i * j) for i in v] for j in v])
    if np.linalg.matrix_rank(q) == q.shape[1]:
        return np.dot(np.linalg.inv(q), [np.sum(im[np.isfinite(im)] * i) for i in v])
    else:
        print("Singular matrix")
        return np.dot(np.linalg.pinv(q), [np.sum(im[np.isfinite(im)] * i) for i in v])
