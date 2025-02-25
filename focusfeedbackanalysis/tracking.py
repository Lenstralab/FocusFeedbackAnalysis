from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from itertools import product
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.gridspec import GridSpec
from ndbioimage import Transforms
from parfor import ParPool, pmap
from scipy.spatial.distance import cdist
from tqdm.auto import trange

from .utilities import TqdmMeter

if __package__ is None or __package__ == "":
    import images
    import localisation
else:
    from . import images, localisation


def interp_nan(g):
    if not g.empty:
        columns = ["x", "y", "s"]
        idx = set()
        for c in columns:
            idx |= set(g[c].index[g[c].apply(np.isnan)])
        for i in idx:
            time = g.loc[i, "T"]
            q = g.loc[i, "particle"]
            steps = 5
            txy = g.query(f"{time}-{steps}<T<{time}+{steps} & T!={time} & particle=={q}")[["T", *columns]]
            if len(txy) > 1:
                for c in columns:
                    g.loc[i, c] = np.dot(np.polyfit(txy["T"], txy[c], 1), (time, 1))
            else:
                raise Exception("Too many missing frames.")
            g.loc[i, "link"] = 4


def localize_main(master, im, transform, cnamelist, mchannel, dist, cylchannels, piezoval, timeval, pool=True):
    """Completes tracks in master"""
    finals = []
    for ch in mchannel:
        m = master.query(f"C == {ch}").copy()
        m = complete_track(m, im, transform, cnamelist, cylchannels, piezoval, timeval, dist, pool=pool)
        # keep only complete tracks
        p = []
        d = done(m, im.shape["t"] - 1, set(filter_stubs(m.query("particle>=0"), im.shape["t"] - 2)["particle"]))
        final = pandas.DataFrame()
        for k, v in d.items():
            if v:
                p.append(k)
                final = pandas.concat((final, m.query(f"particle=={k}")))
        interp_nan(final)
        finals.append(final)
    return pandas.concat(finals, ignore_index=True)


def localise_sm(master, im, transform, slchannel, cylchannels, piezoval, timeval, dist=10):
    """Localizes points dist around master in slave channel, mode: combine, each"""
    if master.empty:
        return pandas.DataFrame(np.zeros((0, len(master.columns))), master.columns)
    else:
        master = master.copy()
        slave = []
        for sch, particle in product(slchannel, master["particle"].unique()):
            s0 = master.query(f"particle == {particle}")
            for T in s0["T"].unique():
                s1 = s0.query(f"T == {T}")
                s = s1.iloc[[0]].copy()
                s["s"] = im.sigma[sch] * np.mean([row["s"] / im.sigma[int(row["C"])] for _, row in s1.iterrows()])
                s["C"] = sch
                s[["x", "y"]] = s1[["x", "y"]].mean().to_numpy()
                s["e"] = np.exp(np.log(s["e"]).mean())
                slave.append(s)

        slave = pandas.concat(slave, ignore_index=True)
        slave = slave.convert_dtypes(True, False, False, False)

        def fun(c, im, time, dist):  # noqa
            index, h = c
            h = h.copy()
            im_a = np.asarray(im(int(h["C"]), int(h["Z"]), int(h["T"])))
            f = forced_localise(h, im_a, None, im.sigma, time, im.cnamelist, dist)
            f = loc(f, im_a, im.sigma[int(h["C"])], time, im.cnamelist, False, dist, True, True, True)
            f.index = [index]
            f["s_ini"] = float(h["s_ini"])
            f["i_ini"] = float(h["i_ini"])
            f["o_ini"] = float(h["o_ini"])
            f["link"] = int(h["link"])
            return f

        g = pandas.concat(
            pmap(fun, slave.iterrows(), (im, transform, dist), desc="Finding slave molecules", total=len(slave)),
            axis=0,
        ).sort_values(by=["C", "T", "particle"])
        g = localisation.insert_z(g, im, cylchannels, piezoval, timeval)
        g = localisation.attach_units(g, im.pxsize_um)
        return pandas.DataFrame(g, columns=master.columns)


def plot_track(f, channel=None):
    plt.figure(figsize=(6, 4))
    if channel is None:
        particles = f["particle"].unique()
    else:
        particles = f.query(f"C=={channel}")["particle"].unique()
    for particle in particles:
        if channel is None:
            time = f.query(f"particle=={particle}").sort_values("T")[["T", "x"]].to_numpy()
        else:
            time = f.query(f"C=={channel} & particle=={particle}").sort_values("T")[["T", "x"]].to_numpy()
        ss = []
        for t in time:
            while (s := len(ss)) < t[0]:
                ss.append([s, np.nan])
            ss.append(t)
        if particle == 0:
            plt.plot(*np.array(ss).T, ".")
        else:
            plt.plot(*np.array(ss).T, ".-")
        plt.legend(particles)


def complete_track(
    m, im, transform, cnamelist, cylchannels, piezoval, timeval, dist=None, min_len=None, pool=False, pl=False
):
    """Fills in the gaps in DataFrame m, optionally uses DataFrame k for help"""
    m = m.copy()
    if dist is None:
        dist = (0, 10)

    min_len = min_len or max(int(np.max(list(tracklen(m.query("particle>=0")).values()), initial=0) / 2), 2)

    m = pre_link(m, min_len)
    m = duplicates(m, min_len)

    min_len = min(min_len, int(np.max(list(tracklen(m.query("particle>=0")).values()), initial=0)) - 1)
    p = set(filter_stubs(m.query("particle>=0"), min_len)["particle"])
    d = done(m, im.shape["t"] - 1, p)
    i = im.shape["t"]
    with TqdmMeter(total=len(p) * im.shape["t"]) as bar:
        # extend tracks to the beginning and end of time
        bar.set_description("Appending to ends of tracks")
        bar.n = len(filter_stubs(m.query("particle>=0"), min_len))
        m["link"] = 0

        def fun(c, *args):
            return forced_localise(*c, *args)

        steps = 10  # how many frames to look for-/backward

        with ExitStack() as stack:
            if pool:
                print("starting pool")
                parpool = ParPool(fun, (transform, cnamelist, dist, steps), n_processes=2)  # type: ignore
                stack.enter_context(parpool)
            else:
                parpool = None

            while not all(d.values()) and i > 0:
                # print(f'0: {i}')
                i -= 1
                for q in p:
                    if q in set(m["particle"]) and not d[q]:
                        m = link(m, im, transform, cnamelist, q, min_len, steps, dist[1], fw=True, pl=pl, pool=parpool)
                        m = link(
                            m, im, transform, cnamelist, q, min_len, steps, dist[1], fw=False, pl=pl, pool=parpool
                        )
                    # plot_track(m)
                p = set(filter_stubs(m.query("particle>=0"), min_len)["particle"])
                d = done(m, im.shape["t"] - 1, p)
                bar.n = len(filter_stubs(m.query("particle>=0"), min_len))
                bar.total = len(p) * im.shape["t"]

            if pool:
                for fw in (True, False):
                    if fw in parpool:
                        f = parpool[fw]
                        idx = f.index[0]
                        m.loc[idx] = f.loc[idx]

            if i == 0:
                print("i=0")

            # fill gaps
            bar.set_description("Filling gaps in tracks")
            i = im.shape["t"]
            g = gaps(m, im.shape["t"] - 1, min_len)
            while any(g.values()) and i > 0:
                # print(f'1: {i}')
                i -= 1
                p = g.keys()
                for q in p:
                    if not m.query(f"particle=={q}").empty:
                        dtype_check(m)
                        # link forwards
                        missing = set(range(im.shape["t"])) - set(m.query(f"particle=={q}")["T"])
                        if missing:
                            m = link(
                                m,
                                im,
                                transform,
                                cnamelist,
                                q,
                                min_len,
                                steps,
                                dist[1],
                                fw=True,
                                next_time=min(missing) - 1,
                                pl=pl,
                                pool=parpool,
                            )
                            # plot_track(m)
                        # link backwards
                        missing = set(range(im.shape["t"])) - set(m.query(f"particle=={q}")["T"])
                        if missing:
                            m = link(
                                m,
                                im,
                                transform,
                                cnamelist,
                                q,
                                min_len,
                                steps,
                                dist[1],
                                fw=False,
                                next_time=max(missing) + 1,
                                pl=pl,
                                pool=parpool,
                            )
                            # plot_track(m)
                g = gaps(m, im.shape["t"] - 1, min_len)
                bar.n = len(filter_stubs(m.query("particle>=0"), min_len))
                bar.total = len(p) * im.shape["t"]

            if pool:
                for fw in (True, False):
                    if fw in parpool:
                        f = parpool[fw]
                        idx = f.index[0]
                        m.loc[idx] = f.loc[idx]

            if i == 0:
                print("i=0 gaps")

        bar.n = len(filter_stubs(m.query("particle>=0"), min_len))
        bar.total = len(p) * im.shape["t"]

    # m = m.convert_dtypes(True, False, False, False)
    m = m.astype(float)
    n = m.copy().query("link!=0")
    m = m.drop(n.index)

    if len(n):

        def refit(c, im, time):  # noqa
            index, c = c
            h = c.copy()
            f = loc(
                h,
                np.asarray(im(int(h["C"]), int(h["Z"]), int(h["T"]))),
                im.sigma[int(h["C"])],
                time,  # noqa
                im.cnamelist,
                False,
                (0, 0),
                True,
                True,
                True,
            )
            f.index = [index]
            f["s_ini"] = float(h["s_ini"])
            f["i_ini"] = float(h["i_ini"])
            f["o_ini"] = float(h["o_ini"])
            f["link"] = int(h["link"])
            return f

        n0 = pandas.concat(
            pmap(refit, n.iterrows(), (im, transform), desc="Refitting localisations", total=len(n)), axis=0
        ).sort_values(by=["C", "T", "particle"])
        n1 = localisation.insert_z(n0, im, cylchannels, piezoval, timeval)
        n2 = localisation.attach_units(n1, im.pxsize_um)

        return pandas.concat((m, n2), sort=False).sort_values("T")
    else:
        return m


def av_fun(im, c, t, av=5, fw=True):
    av = min(5, av)
    s = 1 if fw else -1
    lb, ub = t, t + s * av
    lb, ub = min((lb, ub)), max((lb, ub))
    lb = np.clip(lb, 0, im.shape["t"] - 1)
    ub = np.clip(ub, 0, im.shape["t"] - 1)
    return im[c, lb : ub + 1].nanmax("t")


def filter_stubs(tracks, threshold=100):
    grouped = tracks.reset_index(drop=True).groupby("particle")
    filtered = grouped.filter(lambda x: x["T"].count() >= threshold)
    return filtered.set_index("T", drop=False)


def duplicates(m, min_len, steps=5):
    p = set(filter_stubs(m.query("particle>=0"), min_len)["particle"])
    for q in p:
        d = [item for item, count in Counter(m.query(f"particle=={q}")["T"]).items() if count > 1]
        for T in d:
            dtype_check(m)
            txy = m.query(f"{T}-{steps}<T<{T}+{steps} & T!={T} & particle=={q}")[["T", "x", "y"]]
            if txy.empty:
                txy = m.query(f"T!={T} & particle=={q}")[["T", "x", "y"]]
                txy = txy.iloc[(txy["T"] - T).abs().argsort()[:2]]
            if len(txy) > 1:
                x = np.dot(np.polyfit(txy["T"], txy["x"], 1), (T, 1))
                y = np.dot(np.polyfit(txy["T"], txy["y"], 1), (T, 1))
            elif len(txy) == 1:
                x = float(txy["x"])
                y = float(txy["y"])
            else:
                x, y = np.nan, np.nan

            n = m.query(f"T=={T} & particle=={q}")
            r2 = np.array((n["x"] - x) ** 2 + (n["y"] - y) ** 2)
            idx = n.iloc[[np.argmin(r2)]].index[0]
            jdx = np.array(n.query(f"index!={idx}").index)
            m.loc[jdx, "particle"] = -1
    return m


def pre_link(m, min_len):
    """Connects tracks of different particles together based on proximity"""

    def av_pos(m):  # noqa
        p = set(m.query("particle>=0")["particle"])
        pos = {}
        for q in p:  # noqa
            n = m.query(f"particle=={q}")
            pos[q] = [n["x"].mean(), n["y"].mean()]
        return pos

    def cond():
        l = max(np.floor(min_len / 4), 10)
        m2 = filter_stubs(m, l)
        if not len(m2):
            return np.array(()), np.array(()), []
        pos = av_pos(m2)
        c = np.array(list(pos.values()))  # noqa
        k = list(pos.keys())  # noqa
        c = cdist(c, c)  # noqa
        for i in range(c.shape[0]):
            c[i, i] = 100
        q = c.flatten()  # noqa
        return q, c, k

    q, c, k = cond()
    j = c.shape[0] + 1
    while len(q[q < 5]) and j > 0:
        j -= 1
        a, b = np.unravel_index(np.argmin(c), c.shape)
        # print(a, b)
        m.loc[m.particle == k[b], "particle"] = k[a]
        q, c, k = cond()
    return m


def tracklen(f):
    p = set(f["particle"])
    l = dict()
    for q in p:
        l[q] = len(f.query(f"particle=={q}"))
    return l


def done(m, mx, p):
    d = {}
    for q in p:
        time = m.query(f"particle=={q}")["T"]
        d[q] = time.min() == 0 and time.max() == mx
    return d


def gaps(m, mx, min_len):
    p = set(filter_stubs(m.query("particle>=0"), min_len)["particle"])
    missing = {}
    for q in p:
        missing[q] = set(range(mx + 1)) - set(m.query(f"particle=={q}")["T"])
    return missing


def loc(f: pandas.DataFrame | pandas.Series, im: np.ndarray, s: float, transform: Transforms, cnamelist: Sequence[str],
        filtr: bool, dist: tuple[int, int] = (0, 15), tilt: bool = False,
        sigma: bool = False, xy: tuple[float, float] = False) -> pandas.DataFrame:
    """f: dataframe, single line, single localisation, xy corrected
    im: uncorrected frame
    T: transform used to correct
    #av: scalar, average over av frames
    filtr: apply gaussian bandpass filter?
    sigma: True; fix sigma at value
    xy: True; fix xy at value

    wp@tl201909
    """

    def guess(im, xy, r):  # noqa
        pm = images.crop(im, (xy[0] - r, xy[0] + r + 1), (xy[1] - r, xy[1] + r + 1))
        return (np.unravel_index(np.argmax(pm), pm.shape) + np.array((xy[1] - r, xy[0] - r)))[::-1]

    if isinstance(f, pandas.DataFrame):
        g = f.iloc[0].copy()
    else:
        g = f.copy()

    n = int(g["particle"])
    c = int(g["C"])
    z = int(g["Z"])
    t = int(g["T"])
    g = transform.inverse.coords_pandas(g.copy(), cnamelist)

    # if int(f['T']) == 0:
    #     g = T.inverse.coords(f)
    x = float(g["x"])
    y = float(g["y"])
    sg = float(g["s"])
    ell = float(g["e"])

    if filtr:
        im = images.gfilter(np.asarray(im), s)

    if not xy:
        x, y = guess(im, (x, y), np.ceil(dist[1] + 1))

    h = pandas.DataFrame({"y": y, "x": x, "C": c, "s": sg, "e": ell, "T": 0, "particle": n}, index=[-1])

    keep = []
    theta = False
    if sigma:
        keep.append("s")
    if xy:
        keep.append("x")
        keep.append("y")

    h = localisation.tpsuperresseq(h, im, {c: s}, theta=theta, tilt=tilt, keep=keep, bar=False)  # type: ignore

    h["C"] = c
    h["Z"] = z
    h["T"] = t
    return transform.coords_pandas(h.copy(), cnamelist)


def link(f, im, transform, cnamelist, q, min_len=50, steps=5, lim=5, fw=True, next_time=None, pl=False, pool=None):
    """f: pandas dataframe with localisations, columns particle, C, Z, T, x, y, s, i, o, e should be present
    im: Imread object without applied transforms
    T: transform describing transformation between channels in im; T = Imread(im, transform=True).transform
    q: which particle to link
    steps: how many frames are considered in various steps to link
    fw: True; link forwards, False; link backwards
    mT: None; link at ends, scalar; find new localisation next to T=mT
    pl: True; verbose

    wp@tl201909
    """

    # f = f.convert_dtypes(True, False, False, False)
    if len(f.query(f"particle=={q}")) < min_len:
        return f

    if not pool is None:
        if fw in pool:
            fs = pool[fw]
            idx = fs.index[0]
            f.loc[idx] = fs.loc[idx]

    # the pool might have changed particle number
    if next_time is not None and f.query(f"particle=={q} & T=={next_time}").empty:
        return f

    f = duplicates(f, min_len)
    n = f.query(f"particle=={q}")
    c = int(n["C"].mode().iloc[0])
    n0 = n.query(f"link<2 & C=={c}")
    sigma = im.sigma
    if not n0.empty:
        sigma[c] = np.nanmean(n0["s"])
    # sigma = im.sigma(c)
    s = 1 if fw else -1

    if next_time is None:
        # ends
        next_time = n["T"].max() if fw else n["T"].min()
        next_time0 = n0["T"].max() if fw else n0["T"].min()

        if not 0 <= next_time + s < im.shape["t"] or n0.empty:
            return f
        txy = n0.query(f"{next_time0}-{3 * steps}<T<{next_time0}+{3 * steps}")[["T", "x", "y", "s"]]
        x = np.nanmean(txy["x"])
        y = np.nanmean(txy["y"])
    else:
        # gaps
        time_prev = n0.query(f"T<={next_time + s}")["T"].max()
        if np.isnan(time_prev):
            time_prev = n0["T"].min()
        time_next = n0.query(f"T>={next_time + s}")["T"].min()
        if np.isnan(time_next):
            time_next = n0["T"].max()
        txy = n0.query(f"{max(0, time_prev - 3 * steps)}<T<{time_next + 3 * steps}")[["T", "x", "y", "s"]]
        if len(txy) > 1:
            x = np.dot(np.polyfit(list(txy["T"]), list(txy["x"]), 1), (next_time + s, 1))
            y = np.dot(np.polyfit(list(txy["T"]), list(txy["y"]), 1), (next_time + s, 1))
        else:
            x = float(txy["x"])
            y = float(txy["y"])
    sgm = np.nanmean(txy["s"])

    if pl:
        print(f"Finding new localisation in frame {next_time + s}, particle {q}, xy: {x}, {y}")

    for i in range(steps):
        # find a previously found closeby particle in another track in the next/previous steps frames
        if next_time + s * (i + 1) >= 0:  # pandas chockes on T==-1
            g = f.query(f"T=={next_time + s * (i + 1)} & (x-{x})**2+(y-{y})**2<{lim} & particle!={q} & link<2")
            # g = f.query(f'T=={mT+s*(i+1)} & (x-{x})**2+(y-{y})**2<{lim} & particle!={q}')
            p = g["particle"].unique()

            if len(p):
                for pr in p:
                    if pr >= 0:
                        f.drop(f.query(f"particle=={pr} & link>1").index, inplace=True)
                        f.loc[f["particle"] == pr, "particle"] = q
                    else:
                        f.loc[g.query("particle<0").index, "particle"] = q
                    if pl:
                        print("Found: ", pr)
                f = duplicates(f, min_len)
                if not f.query(f"T=={next_time + s} & particle=={q}").empty:
                    return f

    # try to find a new localisation in a next/previous frame, first by free fit,
    # if that fails, fix some parameter, etc.
    h = n.query(f"T=={next_time}").copy()
    h["x"] = x
    h["y"] = y
    h["s"] = sgm
    h["T"] += s
    h.index = [f.index.max() + 1]

    if pl:
        print("Searching new in frame: {}, xy: {}, {}".format(h.iloc[0]["T"], h.iloc[0]["x"], h.iloc[0]["y"]))

    channel = int(h["C"].iloc[0])
    zslice = int(h["Z"].iloc[0])
    time = int(h["T"].iloc[0])

    if pool is None:
        fs = forced_localise(
            h,
            im(channel, zslice, time),
            av_fun(im, channel, time, steps, fw),
            sigma,
            transform,
            cnamelist,
            (0, lim),
            steps,
        )
        f = pandas.concat((f, fs), sort=False)
    else:
        pool[fw] = (h, im(channel, zslice, time), av_fun(im, channel, time, steps, fw), sigma)
        f = pandas.concat((f, h), sort=False)

    if pl:
        print("Found new in frame: {}, xy: {}, {}".format(h.iloc[0]["T"], h.iloc[0]["x"], h.iloc[0]["y"]))

    return f


def find_around(master, slave, dist):
    locs = []
    for _, row in master.iterrows():
        time, x, y = row[["T", "x", "y"]]
        time2 = slave.query(f"T=={time}").copy()
        time2["dist2"] = (time2["x"] - x) ** 2 + (time2["y"] - y) ** 2
        locs.append(time2.query(f"dist2<={dist**2}"))
    return pandas.concat(locs).sort_index()


def forced_localise(h: pandas.DataFrame | pandas.Series, im: np.ndarray, jm: np.ndarray, sigma: Sequence[float],
                    transform: Transforms, cnamelist: Sequence[str], dist=(0, 5), steps=1) -> pandas.DataFrame:
    """ jm: steps averaged frames """
    if not isinstance(h, pandas.DataFrame):
        h = pandas.DataFrame(h).T

    def check(f, g, sigma, dist):  # noqa
        # g: check if xy in f close to xy in g
        f, g = f.iloc[0], g.iloc[0]  # noqa
        if not (
            dist[0] ** 2 < (float(f["x"]) - float(g["x"])) ** 2 + (float(f["y"]) - float(g["y"])) ** 2 < dist[1] ** 2
        ):
            return False
        if not (max(sigma / 2, 1) < float(f["s"]) < sigma * 2) or not (0.65 < float(f["e"]) < 1.3):
            return False
        return True

    if jm is None:
        jm = im

    sigma = sigma[int(h["C"].iloc[0])]
    if steps > 1:
        a1 = loc(h, jm, sigma, transform, cnamelist, True, dist)
    else:
        a1 = loc(h, im, sigma, transform, cnamelist, True, dist)
    if check(a1, h, sigma, dist):
        if steps > 1:
            a2 = loc(a1, jm, sigma, transform, cnamelist, True, dist)
        else:
            a2 = a1
        if check(a2, h, sigma, dist):
            f = a2
            route = 0
        elif not (
            dist[0] ** 2 < (float(a2["x"].iloc[0]) - float(h["x"].iloc[0])) ** 2 + (float(a2["y"].iloc[0]) - float(h["y"].iloc[0])) ** 2 < dist[1] ** 2
        ):
            if not (max(sigma / 2, 1) < float(h["s"].iloc[0]) < sigma * 2) or not (0.65 < float(h["e"].iloc[0]) < 1.3):
                f = a1
                route = 2
            else:
                b3 = loc(a1, im, sigma, transform, cnamelist, True, dist, xy=True)
                if check(b3, h, sigma, dist):
                    f = b3
                    route = 1
                else:
                    a2["x"] = a1["x"]
                    a2["y"] = a1["y"]
                    f = a2
                    route = 2
        else:
            b4 = loc(a1, im, sigma, transform, cnamelist, True, dist, sigma=True)
            if check(b4, h, sigma, dist):
                f = b4
                route = 1
            else:
                a2["x"] = a1["x"]
                a2["y"] = a1["y"]
                f = a2
                route = 2
    else:
        if steps > 1:
            b1 = loc(h, jm, sigma, transform, cnamelist, True, dist, sigma=True)  # <-- steps -- av in loc
        else:
            b1 = h
        if check(b1, h, sigma, dist):
            c2 = loc(b1, im, sigma, transform, cnamelist, True, dist, sigma=True)
            if check(c2, h, sigma, dist):
                f = c2
                route = 1
            else:
                f = b1
                route = 2
        else:
            f = h  # this will be refitted later to get the correct result, but keep x, y fixed
            route = 2
    f.index = [h.index[0]]
    f["s_ini"] = float(h["s"].iloc[0])
    f["i_ini"] = float(h["i"].iloc[0])
    f["o_ini"] = float(h["o"].iloc[0])
    f["link"] = route + 1
    return f


def kymograph(im, f, timeval, r=25, s=25):
    """Prints a kymograph for all tracks in f longer than s
    im: Imread object
    f: pandas dataframe with tracks
    r: ROI size around average position
    s: plot only tracks longer than s

    wp@tl201909
    """
    particles = f["particle"].unique()
    tracklengths = [f.query(f"particle=={i}").shape[0] for i in particles]
    particles = particles[np.where([i > s for i in tracklengths])[0]]

    xy = []
    h = []
    c = []
    for particle in particles:
        h.append(f.copy().query(f"particle=={particle}"))
        xy.append(np.round((h[-1]["x"].mean(), h[-1]["y"].mean())).astype(int) + 1)
        c.append(int(h[-1]["C"].mode().iloc[0]))

    lp = len(particles)
    x0 = np.zeros((2 * r + 1, im.shape["t"], lp))
    y0 = np.zeros((2 * r + 1, im.shape["t"], lp))

    for t in trange(im.shape["t"], desc="Calculating kymograph"):
        for idx in range(lp):
            jm = im[c[idx], t, xy[idx][1] - r : xy[idx][1] + r + 1, xy[idx][0] - r : xy[idx][0] + r + 1]
            x0[:, t, idx] = jm.max("y")
            y0[:, t, idx] = jm.max("x")

    lt = 1000
    x = np.zeros((2 * r + 1, lt, lp))
    y = np.zeros((2 * r + 1, lt, lp))

    time = np.linspace(0, timeval[-1] - timeval[0], 1000)
    tv = np.array(timeval) - timeval[0]

    for idx in range(lp):
        for i in range(2 * r + 1):
            x[i, :, idx] = np.interp(time, tv, x0[i, :, idx].squeeze())
            y[i, :, idx] = np.interp(time, tv, y0[i, :, idx].squeeze())

    fig = plt.figure(figsize=(11.69, 8.27))
    # fig = plt.figure(figsize=(25, lp*10))
    if lp:
        gs = GridSpec(2 * lp, 1, figure=fig)

        extent = [0, timeval[-1] - timeval[0], im.pxsize_um * r, -im.pxsize_um * r]
        aspect = 2 * (timeval[-1] - timeval[0]) / (2 * r + 1) / max(lp, 1)

        for idx, particle in enumerate(particles):
            # plt.subplot(2*lp, 1, 2*idx+1)
            fig.add_subplot(gs[2 * idx, 0])
            plt.imshow(x[:, :, idx], extent=extent, aspect=aspect)  # type: ignore
            dx = 100 * h[idx]["dx_um"]
            dx[dx == 0] = np.nan
            plt.plot(h[idx]["t"], h[idx]["x_um"] - im.pxsize_um * xy[idx][0] - dx, "--r")
            plt.plot(h[idx]["t"], h[idx]["x_um"] - im.pxsize_um * xy[idx][0], ".r", alpha=0.35, markersize=3)
            plt.plot(h[idx]["t"], h[idx]["x_um"] - im.pxsize_um * xy[idx][0] + dx, "--r")
            plt.ylim(im.pxsize_um * r, -im.pxsize_um * r)
            plt.xlabel("t")
            plt.ylabel(r"x (μm)")
            plt.title(f"Channel {c[idx]}, particle {particle:.0f}")
            # plt.subplot(2*lp, 1, 2*idx+2)
            fig.add_subplot(gs[2 * idx + 1, 0])
            plt.imshow(y[:, :, idx], extent=extent, aspect=aspect)  # type: ignore
            dy = 100 * h[idx]["dy_um"]
            dy[dy == 0] = np.nan
            plt.plot(h[idx]["t"], h[idx]["y_um"] - im.pxsize_um * xy[idx][1] - dy, "--r")
            plt.plot(h[idx]["t"], h[idx]["y_um"] - im.pxsize_um * xy[idx][1], ".r", alpha=0.35, markersize=3)
            plt.plot(h[idx]["t"], h[idx]["y_um"] - im.pxsize_um * xy[idx][1] + dy, "--r")
            plt.ylim(im.pxsize_um * r, -im.pxsize_um * r)
            plt.xlabel("t (s)")
            plt.ylabel(r"y (μm)")
            plt.title(f"Channel {c[idx]}, particle {particle:.0f}")
    plt.tight_layout()
    return fig


def dtype_check(m):
    for c in m.columns:
        if m[c].dtype == "O":
            raise TypeError(f"Column {c} has object dtype")
