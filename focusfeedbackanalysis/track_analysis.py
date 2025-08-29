from __future__ import annotations

import pickle
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import cached_property
from itertools import permutations, product
from numbers import Number
from pathlib import Path
from pprint import pprint
from traceback import format_exc
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas
import skimage
import trackpy
from colorcet import glasbey
from focusfeedbackgui.cylinderlens import calibrate_z
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from ndbioimage import Imread, find
from ndbioimage.transforms import Transforms
from roifile import roiread
from scipy import ndimage, spatial
from tiffwrite import IJTiffFile
from tllab_common.findcells import findcells
from tllab_common.misc import cprint
from tqdm.auto import trange

from . import images, localisation, tracking
from .livecell_functions import ListFile

A4 = 11.69, 8.27


@dataclass
class TrackAnalysis:
    """
    Analyze a time-lapse to construct tracks. A track is constructed on
    the locations of a label (particle in the label channel) in each
    frame. If the label (or label channel) is absent, the track is
    constructed from the particle in the primary channel. Particles in the
    secondary channel (channel_secondary) are localized on or around the
    location of the label of primay track.

    If not done already for a previous analysis, the analysis will start
    with performing a calibration using z-stacks with beads. The argument
    'bead_files' can be used to give the bead files to be used for the
    calibration. If not specified, files in the same folder as the image
    file starting with 'beads' will be used. The channel configuration of
    these z-stacks should be the same as for the image file on which the
    analysis will be done. The calibration is used to determine the angle
    theta of the ellipses to fit, for a correction on the intensity of
    out-of-focus ellipses, and for registering the channels in the image
    file. The registration can be verified by opening the transform.tif
    file saved in the same folder as the image file. The name transform.tif
    is not unique, so make sure there's only one set of bead files per folder.

    The analysis will proceed with determining a mask in which to track. By
    default, mask_method=("square", 30), the mask will be a 30x30 pixel
    region in the center of the cell. Other options are: ("findcells",
    kwargs_dict), see tllab_common.findcells.findcells for details,
    ("array", numpy array) for a manual mask defined by a numpy array, or
    ("roifile", /path/to/file.roi) for a manual mask save in a Fiji roi
    file.

    Particles in the mask are then detected, localized and linked into
    tracks. Of these tracks (if more than 1), the brightest will be
    selected.

    The analysis can then be saved in computer (pickle) and human-readable
    (tsv) formats. Plots can be saved as pdf's.

    Args:
        image_file: (required) time-lapse image file to analyze
        channel_label: channel index/indices for the label if present
        channel_primary: channel index/indices for the particles to be tracked
        channel_secondary: channel index/indices for the particles whose
            intensity will be measured at positions of primary particles
        dist_channel: maximum distance (pixels) between the label or primary
            particles and secondary particles
        dist_frame: maximum distance (pixels) between the label or primary
            particle in consecutive frames
        path_out: where to save the analysis
        calibration_path: storage for calibration files, so that they can be
            reused
        mask_method: ("square", size), ("findcells", kwargs_dict),
            ("array", numpy array) or ("roifile", /path/to/file.roi)
        bead_files: list of paths to bead files to use for various
            calibrations
        wavelengths: list of emission wavelengths (nm) for each channel
            (e.g. eGFP = 510) in the image file, the list needs to be in
            the same order as the channels in the image file
        colors: colors to be used during plotting, the colors need to be
            in the order label, primary, seccondary
        track3D: whether to use z information during linking
    """

    image_file: Path | str | Imread
    channel_label: int | tuple = None
    channel_primary: int | tuple = None
    channel_secondary: int | tuple = None
    channel_mask: int = 1
    dist_channel: float | tuple[int, float] = 3
    dist_frame: float | tuple[int, float] = 5
    path_out: Path | str = None
    calibration_path: Path | str = None
    mask_method: tuple[str] = ("square", 30)
    bead_files: Sequence[Path | str] = None
    wavelengths: tuple = None
    colors: str = None
    track3D: bool = True

    def __post_init__(self):
        if isinstance(self.channel_label, Number):
            self.channel_label = (self.channel_label,)
        elif isinstance(self.channel_label, list):
            self.channel_label = tuple(self.channel_label)
        elif self.channel_label is None:
            self.channel_label = ()
        if isinstance(self.channel_primary, Number):
            self.channel_primary = (self.channel_primary,)
        elif isinstance(self.channel_primary, list):
            self.channel_primary = tuple(self.channel_primary)
        elif self.channel_primary is None:
            self.channel_primary = ()
        if isinstance(self.channel_secondary, Number):
            self.channel_secondary = (self.channel_secondary,)
        elif isinstance(self.channel_secondary, list):
            self.channel_secondary = tuple(self.channel_secondary)
        elif self.channel_secondary is None:
            self.channel_secondary = ()
        if isinstance(self.dist_channel, Number):
            self.dist_channel = (0, self.dist_channel)
        if isinstance(self.dist_frame, Number):
            self.dist_frame = (0, self.dist_frame)
        if self.bead_files is not None:
            self.bead_files = [Path(bead_file).absolute() for bead_file in self.bead_files]
        else:
            self.bead_files = []

        if isinstance(self.image_file, str | Path):
            self.image_file = Path(self.image_file)
            self.im = Imread(self.image_file, axes="ctyx", dtype=float)
            self.jm = Imread(self.image_file, axes="ctyx", dtype=float).with_transform(bead_files=self.bead_files)
        else:
            self.im = self.image_file.view().with_transform()
            self.jm = self.image_file.view().with_transform(bead_files=self.bead_files)
            self.image_file = Path(self.image_file.path)
        if not self.bead_files:
            self.bead_files = self.jm.transform.get_bead_files(self.jm.path.parent)
        if self.wavelengths is not None:
            self.im.sigma = self.jm.sigma = [
                (w * self.im.ureg.nm / 2 / self.im.NA / self.im.pxsize).magnitude for w in self.wavelengths
            ]

        if self.path_out is None:
            self.path_out = Path(str(self.image_file.parent).replace("data", "analysis")) / self.image_file.name
        elif not Path(self.path_out).is_absolute():
            self.path_out = Path(str(self.image_file.parent).replace("data", "analysis")) / Path(self.path_out)
        else:
            self.path_out = Path(self.path_out)
        self.path_out.mkdir(parents=True, exist_ok=True)

        if self.calibration_path is None:
            self.calibration_path = Path(str(self.image_file.parent).replace("data", "analysis"))
        elif not Path(self.calibration_path).is_absolute():
            self.calibration_path = Path(str(self.image_file.parent).replace("data", "analysis")) / Path(
                self.calibration_path
            )
        else:
            self.calibration_path = Path(self.calibration_path)
        self.calibration_path.mkdir(parents=True, exist_ok=True)

        self.exp_name = self.image_file.stem
        self.file = self.path_out / self.image_file.with_suffix(".pk").name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = format_exc()
        else:
            self.error = None
        self.close()

    @staticmethod
    def load_analysis(pkfile: Path | str) -> dict[str, Any]:
        with pandas.HDFStore(str(Path(pkfile).with_suffix(".pk"))) as store:
            return {key[1:]: store.get(key) for key in store.keys()}

    def save(self) -> None:
        with open(self.file, "wb") as f:
            pickle.dump(self, f)  # type: ignore

    @staticmethod
    def load(file: Path | str) -> TrackAnalysis:
        with open(file, "rb") as f:
            return pickle.load(f)

    @cached_property
    def meant(self) -> np.ndarray:
        return self.im[self.channel_mask].mean(0)

    @cached_property
    def cell_and_nuc(self) -> tuple[np.ndarray, np.ndarray]:
        # Calculate mask using the transformed channel
        if self.mask_method[0] == "findcells":
            print("Localizing cells and nuclei")
            kwargs = self.mask_method[1] if len(self.mask_method) >= 2 else {"ccdist": 25}
            cell, nuc = findcells(self.meant, **kwargs)
        elif self.mask_method[0] == "array":
            cell = nuc = self.mask_method[1]
        elif self.mask_method[0] == "roifile":
            roi = roiread(Path(self.image_file).with_suffix(".roi"))
            yx = np.array(np.meshgrid(np.arange(self.im.shape["x"]), np.arange(self.im.shape["y"]))).reshape((2, -1)).T
            dist = spatial.distance.cdist(yx, roi.coordinates()).min(1).reshape(self.im.shape["yx"])
            nuc = cell = skimage.morphology.flood_fill(
                (dist <= 1).astype(int), tuple(roi.coordinates().mean(0).round().astype(int)), 1
            )
        else:
            cell = np.zeros(self.im.shape["yx"])
            size = self.mask_method[1] // 2
            cell[
                self.im.shape["y"] // 2 - size : self.im.shape["y"] // 2 + size,
                self.im.shape["x"] // 2 - size : self.im.shape["x"] // 2 + size,
            ] = 1
            nuc = cell
        return nuc, cell

    @cached_property
    def nuc(self) -> np.ndarray:
        return self.cell_and_nuc[0]

    @cached_property
    def cell(self) -> np.ndarray:
        return self.cell_and_nuc[1]

    @cached_property
    def cellnr(self) -> int:
        cellpos = images.get_nearest_px_msk(
            self.nuc.astype(float), [int(self.im.shape["x"] / 2), int(self.im.shape["y"] / 2)]
        )
        return int(self.cell[cellpos[0], cellpos[1]])

    @cached_property
    def mask(self) -> list[np.ndarray]:
        n = self.nuc == self.cellnr
        n = ndimage.morphology.binary_dilation(n, images.disk(5)).astype(bool)

        # Transform back to get a mask for the untransformed channel(s)
        mask = []
        for c in range(self.im.shape["c"]):
            m = self.jm.transform[self.jm.cnamelist[self.channel_mask]]
            if c == self.channel_mask:
                mask.append(n)
            else:
                mask.append((m * self.jm.transform[self.jm.cnamelist[c]].inverse).frame(n))
        return mask

    @cached_property
    def theta(self) -> float:
        # TODO: use (track, detector) in stead of channels
        theta = []
        for file in self.bead_files:
            calib_file = self.calibration_path / file.with_suffix(".cyllens_calib.pk").name
            if calib_file.exists():
                with open(calib_file, "rb") as f:
                    res = pickle.load(f)
            else:
                res = calibrate_z(file, self.wavelengths, self.cyllenschannels, path=self.calibration_path / file.stem)
                with open(calib_file, "wb") as f:
                    pickle.dump(res, f)  # type: ignore
            theta.append(res["theta"])
        return np.mean(theta) if theta else 0

    @cached_property
    def detected(self) -> pandas.DataFrame:
        print("Localizing interesting points in the cell in the center")
        return localisation.detect_points(self.im, self.im.sigma, self.mask[0])  # type: ignore

    @cached_property
    def cyllenschannels(self) -> list[int]:
        return [
            i
            for i, channel in enumerate(self.im.ome.images[0].pixels.channels)
            if self.im.cyllens[int(channel.detector_settings.id[-1])] != "None"  # type: ignore
        ]

    @cached_property
    def localized(self) -> tuple[pandas.DataFrame, ...]:
        print(
            f"Fitting gaussians to filtered images, ellipticity in channels {self.cyllenschannels}"
            f" with theta={self.theta}"
        )
        unfiltered, filtered, refitted, refiltered, warped = [], [], [], [], []
        for c in range(self.im.shape["c"]):
            if c in (*self.channel_label, *self.channel_primary, *self.channel_secondary):
                f0 = self.detected.query(f"C=={c}").copy()
                self.im.frame_decorator = lambda im, frame, c, z, t: images.gfilter(frame, im.sigma[c])  # noqa
                if c in self.cyllenschannels:
                    g0 = localisation.tpsuperresseq(
                        f0,
                        self.im,
                        self.im.sigma,  # type: ignore
                        theta=self.theta,
                        tilt=False,
                        desc=f"Fitting channel {c} without tilt and theta = {self.theta:.2f}",
                    ).dropna()
                    self.im.frame_decorator = None
                    g0["dr"] = np.sqrt(g0["dx"] ** 2 + g0["dy"] ** 2)
                    g1 = g0.query(
                        f"dr < 0.5 & {self.im.sigma[c] / 2} < s <{self.im.sigma[c] * 2} "  # type: ignore
                        "& 0.65 < e < 1.3 & R2 > 0"
                    )
                    g2 = localisation.tpsuperresseq(
                        g1,
                        self.im,
                        self.im.sigma,  # type: ignore
                        theta=self.theta,
                        tilt=True,
                        keep=["x", "y", "s", "e"],
                        desc=f"Fitting channel {c} with tilt and theta = {self.theta:.2f}",
                    ).dropna()
                    g2["dr"] = np.sqrt(g2["dx"] ** 2 + g2["dy"] ** 2)
                    g3 = g2.query("ds < 2 * s & di < 2 * i & de < 2 * e & 0.65 < e < 1.3 & R2 > 0")
                else:
                    g0 = localisation.tpsuperresseq(
                        f0,
                        self.im,
                        self.im.sigma,  # type: ignore
                        theta=False,
                        tilt=False,
                        desc=f"Fitting channel {c} without tilt or ellipicity",
                    ).dropna()
                    self.im.frame_decorator = None
                    g0["dr"] = np.sqrt(g0["dx"] ** 2 + g0["dy"] ** 2)
                    g1 = g0.query(
                        f"dr < 0.5 & {self.im.sigma[c] / 2} < s < {self.im.sigma[c] * 2} "  # type: ignore
                        "& R2 > 0"
                    )
                    g2 = localisation.tpsuperresseq(
                        g1,
                        self.im,
                        self.im.sigma,  # type: ignore
                        theta=False,
                        tilt=True,
                        keep=["x", "y", "s"],
                        desc=f"Fitting channel {c} with tilt, without ellipicity",
                    ).dropna()
                    g2["dr"] = np.sqrt(g2["dx"] ** 2 + g2["dy"] ** 2)
                    g3 = g2.query("ds < 2 * s & di < 2 * i & R2 > 0 & i > 300")
                g4 = localisation.merge_dataframes(g1, g3)
                g4["x_nt"], g4["y_nt"] = g4["x"], g4["y"]
                if not g4.empty:
                    g4 = self.jm.transform.coords_pandas(g4, self.im.cnamelist)  # type: ignore
                unfiltered.append(g0)
                filtered.append(g1)
                refitted.append(g2)
                refiltered.append(g3)
                warped.append(g4)
        unfiltered = pandas.concat(unfiltered).sort_index()
        filtered = pandas.concat(filtered).sort_index()
        refitted = pandas.concat(refitted).sort_index()
        refiltered = pandas.concat(refiltered).sort_index()

        # from this point on we work with world coordinates,
        # when we need to fit a part of the image again we temporarily convert back to image coordinates
        warped = pandas.concat(warped).sort_index()
        warped = localisation.insert_z(warped, self.im, self.cyllenschannels, self.piezoval, self.timeval)
        warped = localisation.attach_units(warped, self.im.pxsize_um)
        return unfiltered, filtered, refitted, refiltered, warped

    @cached_property
    def piezoval(self) -> pandas.DataFrame:
        """gives the height of the piezo and focus motor, only available when CylLensGUI was used"""

        # Or maybe in an extra '.pzl' file
        m = self.im.extrametadata
        if "Columns" in self.im.extrametadata:
            columns = self.im.extrametadata["Columns"]
            ptime_col = columns.index("frame")
            pval_col = columns.index("piezoPos")
            sval_col = columns.index("focusPos")
        else:
            ptime_col, pval_col, sval_col = 0, 1, 2

        if m is not None and "p" in m:
            q = np.array(m["p"])
            if not len(q.shape):
                q = np.zeros((1, 3))

            ptime = [int(i) for i in q[:, ptime_col]]
            pval = [float(i) for i in q[:, pval_col]]
            sval = [float(i) for i in q[:, sval_col]]
        else:
            ptime = []
            pval = []
            sval = []

        df = pandas.DataFrame(columns=["frame", "piezoZ", "stageZ"])
        df["frame"] = ptime
        df["piezoZ"] = pval
        ref_z = (self.im.ome.images[0].pixels.planes[0].position_z / self.im.ureg.um).magnitude  # type: ignore
        df["stageZ"] = np.array(sval) - np.array(pval) - ref_z

        # remove duplicates
        df = df[~df.duplicated("frame", "last")]
        return df

    @cached_property
    def timeval(self) -> list[float]:
        t0 = find(self.im.ome.images[0].pixels.planes, the_c=0, the_z=0, the_t=0).delta_t
        return [
            find(self.im.ome.images[0].pixels.planes, the_c=0, the_z=0, the_t=t).delta_t - t0
            for t in range(self.im.shape["t"])
        ]

    @cached_property
    def unfiltered(self) -> pandas.DataFrame:
        return self.localized[0]

    @cached_property
    def filtered(self) -> pandas.DataFrame:
        return self.localized[1]

    @cached_property
    def refitted(self) -> pandas.DataFrame:
        return self.localized[2]

    @cached_property
    def refiltered(self) -> pandas.DataFrame:
        return self.localized[3]

    @cached_property
    def warped(self) -> pandas.DataFrame:
        return self.localized[4]

    @cached_property
    def i_calibration(self) -> Optional[tuple[float, float, float, float]]:
        return (
            localisation.calibrate_intensity(
                self.im, self.bead_files, self.calibration_path, self.cyllenschannels, self.piezoval, self.timeval
            )
            if self.bead_files
            else None
        )

    @cached_property
    def linked(self) -> pandas.DataFrame:
        print("Tracking points")
        linked = []
        particle = 0
        for c in (*self.channel_label, *self.channel_primary, *self.channel_secondary):
            if c is not None:
                tl = self.warped.query(f"C=={c}").copy()
                if len(tl):
                    cols = ["x_um", "y_um", "z_um"] if c in self.cyllenschannels and self.track3D else ["x_um", "y_um"]
                    tl = tl.sort_values("T")
                    tl["z_um"] = tl["z_um"].ffill()
                    tl = tl.dropna(subset=["T"] + cols)
                    loc = trackpy.link_df(
                        tl,
                        search_range=self.dist_frame[1] * self.im.pxsize_um,
                        memory=10,
                        pos_columns=cols,
                        t_column="T",
                    )
                    loc.loc[loc["particle"] == 0, "particle"] = loc["particle"].max() + 1
                    # TODO: replace particle 0
                else:
                    loc = pandas.DataFrame(np.zeros((0, len(tl.columns) + 1)), columns=list(tl.columns) + ["particle"])
                if c in self.cyllenschannels and self.i_calibration:
                    print("Correcting peak intensities based on distance from focus")
                    loc = localisation.correct_intensity(loc, *self.i_calibration, 0.1)
                    # Return uncorrected i_peak to frames where the correction could not be done
                    ln = np.where(np.isnan(loc["i_peak"]))[0]  # type: ignore
                    if len(ln):
                        idx = loc.iloc[ln].index.tolist()
                        loc.loc[idx, "i_peak"] = loc.loc[idx, "i_peak_uc"]
                else:
                    loc["i_peak_uc"], loc["di_peak_uc"] = np.nan, np.nan
                if len(loc):
                    loc["particle"] += particle
                    particle = loc["particle"].max() + 1
                    linked.append(loc)
        return pandas.concat(linked).sort_index()

    @cached_property
    def filled(self) -> pandas.DataFrame:
        if self.channel_label:
            label = tracking.localize_primary(
                self.linked,
                self.im,
                self.jm.transform,
                self.im.cnamelist,  # type: ignore
                self.channel_label,
                self.dist_frame,
                self.cyllenschannels,
                self.piezoval,
                self.timeval,
                pool=False,
            )
            if len(self.channel_label) > 1:
                label = self.link_channels(label, 10, self.channel_label)
            if self.channel_primary:
                primary = tracking.localise_secondary(
                    label,
                    self.im,
                    self.jm.transform,
                    self.channel_primary,
                    self.cyllenschannels,
                    self.piezoval,
                    self.timeval,
                    self.dist_channel,
                )
                if self.channel_secondary:
                    secondary = tracking.localise_secondary(
                        primary,
                        self.im,
                        self.jm.transform,
                        self.channel_secondary,
                        self.cyllenschannels,
                        self.piezoval,
                        self.timeval,
                        self.dist_channel,
                    )
                else:
                    secondary = None
            elif self.channel_secondary:
                primary = None
                secondary = tracking.localise_secondary(
                    label,
                    self.im,
                    self.jm.transform,
                    self.channel_secondary,
                    self.cyllenschannels,
                    self.piezoval,
                    self.timeval,
                    self.dist_channel,
                )
            else:
                primary, secondary = None, None
        else:
            label = None
            primary = tracking.localize_primary(
                self.linked,
                self.im,
                self.jm.transform,
                self.im.cnamelist,  # type: ignore
                self.channel_primary,
                self.dist_frame,
                self.cyllenschannels,
                self.piezoval,
                self.timeval,
                pool=False,
            )
            if self.channel_secondary:
                secondary = tracking.localise_secondary(
                    primary,
                    self.im,
                    self.jm.transform,
                    self.channel_secondary,
                    self.cyllenschannels,
                    self.piezoval,
                    self.timeval,
                    self.dist_channel,
                )
            else:
                secondary = None
        return pandas.concat([i for i in (label, primary, secondary) if i is not None], ignore_index=True)

    @cached_property
    def loc_label(self) -> pandas.DataFrame:
        return self.filled.query(f"C in {self.channel_label}")

    @cached_property
    def loc_primary(self) -> pandas.DataFrame:
        return self.filled.query(f"C in {self.channel_primary}")

    @cached_property
    def loc_secondary(self) -> pandas.DataFrame:
        return self.filled.query(f"C in {self.channel_secondary}")

    @cached_property
    def brightest(self) -> pandas.DataFrame:
        # keep brightest only
        if self.channel_label:
            localizations = self.loc_label
            channels = self.channel_label
        else:
            localizations = self.loc_primary
            channels = self.channel_primary
        self.particles = []
        for channel in channels:
            loc = localizations.query(f"C==@channel", local_dict=dict(channel=channel))
            p = list(loc["particle"].unique())
            i_peak = [loc.query(f"link==0 & particle=={q}")["i_peak"].sum() for q in p]
            self.particles.append(p[np.argmax(i_peak)])
        if self.channel_label:
            label = self.loc_label.query(f"particle in @particles", local_dict=dict(particles=self.particles))
        else:
            label = None
        primary = self.loc_primary.query(f"particle in @particles", local_dict=dict(particles=self.particles))
        secondary = self.loc_secondary.query(f"particle in @particles", local_dict=dict(particles=self.particles))
        return pandas.concat([i for i in (label, primary, secondary) if i is not None], ignore_index=True)

    @staticmethod
    def link_channels(df: pandas.DataFrame, distance: float, channels: Sequence[int]) -> pandas.DataFrame:
        df = df.copy()
        if len(channels) != 2:
            raise NotImplementedError(f"# channels: {len(channels)} not implemented")
        particles = [df.query(f"C == {C}")["particle"].unique() for C in channels]
        distances = np.zeros([len(particle) for particle in particles])
        for p in product(*[enumerate(particle) for particle in particles]):
            i, q = zip(*p)
            distances[i] = np.sqrt(
                np.sum(
                    (
                        df.query(f"C=={channels[0]} & particle=={q[0]}")[["x", "y"]].to_numpy()
                        - df.query(f"C=={channels[1]} & particle=={q[1]}")[["x", "y"]].to_numpy()
                    )
                    ** 2,
                    1,
                )
            ).mean()
        idx = [np.where(distances.min(i) < distance)[0] for i in (1, 0)]
        particles = particles[0][idx[0]], particles[1][idx[1]]
        distances = distances[idx[0]][:, idx[1]]
        n = min(*[len(i) for i in particles])
        a = {
            k: np.sqrt(np.sum(distances[tuple([i[:n] for i in k])] ** 2))
            for k in product(*[permutations(range(s)) for s in distances.shape])
        }
        assignments = min(a, key=a.get)
        assignments = [
            particle[np.array(assignment)[:n]].astype(int) for assignment, particle in zip(assignments, particles)
        ]
        for i, j in zip(*assignments):
            df.loc[df.query(f"C==1 & particle=={j}").index, "particle"] = i
        return df

    @cached_property
    def loc_label_brightest(self) -> pandas.DataFrame:
        return self.brightest.query(f"C in {self.channel_label}").sort_values("T")

    @cached_property
    def loc_primary_brightest(self) -> pandas.DataFrame:
        return self.brightest.query(f"C in {self.channel_primary}").sort_values("T")

    @cached_property
    def loc_secondary_brightest(self) -> pandas.DataFrame:
        return self.brightest.query(f"C in {self.channel_secondary}").sort_values("T")

    @cached_property
    def background(self) -> pandas.DataFrame:
        cprint("<Fitting background.:b.b>")
        return self.get_background(self.data, self.im, self.jm.transform, self.im.cnamelist)  # type: ignore

    @cached_property
    def background_label(self) -> pandas.DataFrame:
        return self.background.query(f"C in {self.channel_label}")

    @cached_property
    def background_primary(self) -> pandas.DataFrame:
        return self.background.query(f"C in {self.channel_primary}")

    @cached_property
    def background_secondary(self) -> pandas.DataFrame:
        return self.background.query(f"C in {self.channel_secondary}")

    def plot(self) -> None:
        with PdfPages(self.path_out / f"{self.exp_name}_cellnr_{self.cellnr}_trk_results_trace.pdf") as pdf:
            self.plot_info(pdf)
            self.plot_mask(pdf)
            self.plot_localizations(pdf)
            self.plot_kymograph(pdf)
            self.plot_traces(pdf)
            self.plot_traces_extra(pdf)
            self.plot_distances(pdf)

    def plot_info(self, pdf: PdfPages = None) -> None:
        fig = plt.figure(figsize=A4)
        plt.text(0.05, 0.5, self.im.__repr__().replace("#" * 106, ""), va="center")
        plt.axis("off")
        if pdf:
            pdf.savefig(fig)
            plt.close(fig)

    def plot_mask(self, pdf: PdfPages = None) -> None:
        fig = plt.figure(figsize=A4)
        gs = GridSpec(2, 3, figure=fig)

        cmap = glasbey.copy()
        cmap.insert(0, "#ffffff")
        cmap = ListedColormap(cmap)

        fig.add_subplot(gs[0, 0])
        plt.imshow(self.im[self.channel_primary[0], self.im.shape["t"] // 2], cmap="gray")
        plt.title(f"primary channel ({self.channel_primary[0]}), t = {self.im.shape['t'] // 1}")
        fig.add_subplot(gs[1, 0])
        plt.imshow(self.meant, cmap="gray")
        plt.title(f"mean <t> image in mask channel ({self.channel_mask})")
        fig.add_subplot(gs[0, 1])
        plt.imshow(self.cell, cmap=cmap, vmax=256)
        plt.title("cell mask or ROI")
        for i in range(int(self.cell.max())):
            try:
                plt.text(
                    *[int(j.mean()) for j in np.where(self.cell == (i + 1))][::-1],
                    f"{i + 1}",  # type: ignore
                    va="center",
                    ha="center",
                    color="w",
                )
            except Exception:  # noqa
                pass
        fig.add_subplot(gs[1, 1])
        plt.imshow(self.nuc, cmap=cmap, vmax=256)
        plt.title("nucleus mask or ROI")
        for i in range(int(self.nuc.max())):
            try:
                plt.text(
                    *[int(j.mean()) for j in np.where(self.nuc == (i + 1))][::-1],
                    f"{i + 1}",  # type: ignore
                    va="center",
                    ha="center",
                    color="w",
                )
            except Exception:  # noqa
                pass
        fig.add_subplot(gs[0, 2])
        plt.imshow(self.meant, cmap="gray")
        plt.title(f"mean <t> image in mask channel ({self.channel_mask})")
        nc = images.approxcontour(self.nuc > 0)
        plt.plot(nc[:, 0], nc[:, 1], "--r")
        cc = images.approxcontour(self.cell > 0)
        plt.plot(cc[:, 0], cc[:, 1], "--b")

        if len(self.mask) == 1:
            msk = np.zeros(self.im.shape["yx"])
        else:
            msk = np.zeros((self.im.shape["y"], self.im.shape["x"], 3))

        for i in range(min(len(self.mask), 3)):
            msk[:, :, i] = self.mask[i]
        fig.add_subplot(gs[1, 2])
        plt.imshow(msk)
        plt.title("mask, green: mask channel\nred: warped channels")
        if pdf:
            pdf.savefig(fig)
            plt.close(fig)

    def plot_localizations(self, pdf: PdfPages = None) -> None:
        fig = plt.figure(figsize=A4)
        gs = GridSpec(1, len((*self.channel_label, *self.channel_primary, *self.channel_secondary)) + 1, figure=fig)
        fig.add_subplot(gs[0, 0])
        localisation.plot_localisations(self.linked, self.jm, self.channel_mask, 0, self.jm.shape["t"] // 2, False)
        c = 1
        for ch in (*self.channel_label, *self.channel_primary, *self.channel_secondary):
            if not (tp := self.linked.query(f"C == {ch}").dropna()).empty:
                fig.add_subplot(gs[0, c])
                for _, group in tp.groupby("particle"):
                    plt.plot(*group[["x_um", "y_um"]].to_numpy().T, "-o", ms=3)
                plt.gca().invert_yaxis()
                plt.axis("equal")
                plt.xlabel(r"x (μm)")
                plt.ylabel(r"y (μm)")
                plt.title(f"tracks in channel {ch}")
                c += 1
        plt.tight_layout()
        if pdf:
            pdf.savefig(fig)
            plt.close(fig)

    def plot_distances(self, pdf: PdfPages = None) -> None:
        if len(self.channel_label) == 2:
            fig = plt.figure(figsize=A4)
            gs = GridSpec(1, 2, figure=fig)
            fig.add_subplot(gs[0, 0])
            dist = (
                self.data.query(f"C=={self.channel_label[1]}").sort_values("T")[["x_um", "y_um"]].to_numpy()
                - self.data.query(f"C=={self.channel_label[0]}").sort_values("T")[["x_um", "y_um"]].to_numpy()
            ).T
            plt.plot(*dist, "-k", linewidth=1, zorder=1)
            s = plt.scatter(
                *dist, c=self.data.query(f"C=={self.channel_label[0]}").sort_values("T")["t"], zorder=2, cmap="plasma"
            )
            c = plt.colorbar(s)
            c.set_label("time (s)")
            plt.xlabel(r"distance between labels x (μm)")
            plt.ylabel(r"distance between labels y (μm)")

            fig.add_subplot(gs[0, 1])
            dist = np.sqrt(
                (
                    (
                        self.data.query(f"C=={self.channel_label[0]}").sort_values("T")[["x_um", "y_um"]].to_numpy()
                        - self.data.query(f"C=={self.channel_label[1]}").sort_values("T")[["x_um", "y_um"]].to_numpy()
                    )
                    ** 2
                ).sum(1)
            )
            plt.plot(self.data.query(f"C=={self.channel_label[0]}").sort_values("T")["t"], dist, "k")
            plt.xlabel("time (s)")
            plt.ylabel(r"distance between labels (μm)")
            if pdf:
                pdf.savefig(fig)
                plt.close(fig)

    def save_tracking(self) -> None:
        self.show_tracking_ncolor()
        self.save_results()
        self.write_files()

    @cached_property
    def data(self) -> pandas.DataFrame:
        return pandas.concat(
            [
                i
                for i in (self.loc_label_brightest, self.loc_primary_brightest, self.loc_secondary_brightest)
                if i is not None
            ],
            ignore_index=True,
        )

    @cached_property
    def channel(self) -> tuple[int, ...]:
        return *self.channel_label, *self.channel_primary, *self.channel_secondary

    @cached_property
    def color(self) -> str:
        w = [self.im.laserwavelengths[c][0] for c in self.channel]
        if self.colors is None:
            return "".join(["rgbcmy"[i] for i in np.argsort(w)[::-1]])
        else:
            return self.colors

    def plot_kymograph(self, pdf: PdfPages = None) -> None:
        for channel in self.channel:
            fig = tracking.kymograph(self.jm, self.data.query(f"C == {channel}"), self.timeval)
            if pdf:
                pdf.savefig(fig)
                plt.close(fig)

    def plot_traces(self, pdf: PdfPages = None) -> None:
        channels = self.channel_label + self.channel_primary + self.channel_secondary
        fig = self.show_data(self.data, self.color, channels)
        if pdf:
            pdf.savefig(fig)
            plt.close(fig)

    def plot_traces_extra(self, pdf: PdfPages = None) -> None:
        fig = plt.figure(figsize=A4)
        plt.suptitle(f"particles: {[int(p) for p in self.particles]}")
        gs = GridSpec(5, 2, figure=fig)

        gs0 = GridSpecFromSubplotSpec(1, 5, gs[0, 0])
        fig.add_subplot(gs0[0, :4])
        t = np.array(self.timeval)
        dt = np.diff(t)
        ax = plt.plot(t[1:] - t[0], dt)
        plt.xlim(0, t.max() - t.min())
        plt.text(
            0.95,
            0.95,
            rf"<$\Delta$t> = {dt.mean():.2f} $\pm$ {dt.std():.2f}",
            transform=ax[0].axes.transAxes,
            ha="right",
            va="top",
        )
        plt.xlabel("time (s)")
        plt.ylabel("timeinterval (s)")
        ylim = plt.ylim()
        fig.add_subplot(gs0[0, 4])
        plt.hist(dt, 100, orientation="horizontal")
        plt.ylim(ylim)
        plt.tick_params(left=False, labelleft=False)

        fig.add_subplot(gs[0, 1])
        frames = self.piezoval["frame"].to_numpy()
        idx = (0 <= frames) & (frames < self.im.shape["t"])
        p = np.array(self.piezoval["piezoZ"][idx])
        pt = t[self.piezoval["frame"][idx]] - t[0]
        plt.plot(pt, 1000 * p)
        plt.xlim(0, t.max() - t.min())
        plt.xlabel("time (s)")
        plt.ylabel("piezo position (nm)")

        fig.add_subplot(gs[1, 0])
        for channel, c in zip(self.channel, self.color):
            d = self.data.query(f"C == {channel}")
            plt.plot(d["t"], 1000 * d["s_um"], c)

        plt.xlim(0, self.data["t"].max())
        plt.xlabel("time (s)")
        plt.ylabel("sigma (nm)")

        fig.add_subplot(gs[1, 1])
        for channel, c in zip(self.channel, self.color):
            d = self.data.query(f"C == {channel}")
            plt.plot(d["t"], d["o"], c)
        plt.xlim(0, self.data["t"].max())
        plt.xlabel("time (s)")
        plt.ylabel("offset")

        fig.add_subplot(gs[2, 0])
        for channel, c in zip(self.channel, self.color):
            d = self.data.query(f"C == {channel}")
            plt.plot(d["t"], d["e"], c)
        plt.xlim(0, self.data["t"].max())
        plt.xlabel("time (s)")
        plt.ylabel("ellipticity")

        fig.add_subplot(gs[2, 1])
        for channel, c in zip(self.channel, self.color):
            d = self.data.query(f"C == {channel}")
            plt.plot(d["t"], d["z_um"], c)
        plt.xlim(0, self.data["t"].max())
        plt.xlabel("time (s)")
        plt.ylabel(r"z ($\mathrm{\mu}$m)")

        fig.add_subplot(gs[3, 0])
        for channel, c in zip(self.channel, self.color):
            d = self.data.query(f"C == {channel}")
            plt.plot(d["t"], d["i_peak"], c)
        plt.xlim(0, self.data["t"].max())
        plt.xlabel("time (s)")
        plt.ylabel("peak intensity")

        fig.add_subplot(gs[3, 1])
        for channel, c in zip(self.channel, self.color):
            d = self.data.query(f"C == {channel}")
            plt.plot(d["t"], d["i"], c)
        plt.xlim(0, self.data["t"].max())
        plt.xlabel("time (s)")
        plt.ylabel("integrated intensity")

        fig.add_subplot(gs[4, 0])
        for channel, c in zip(self.channel, self.color):
            d = self.background.query(f"C == {channel}")
            plt.plot(d["t"], d["i_peak"], c)
        plt.xlim(0, self.data["t"].max())
        plt.xlabel("time (s)")
        plt.ylabel("backgound\npeak intensity")

        fig.add_subplot(gs[4, 1])
        for channel, c in zip(self.channel, self.color):
            d = self.background.query(f"C == {channel}")
            plt.plot(d["t"], d["i"], c)
        plt.xlim(0, self.data["t"].max())
        plt.xlabel("time (s)")
        plt.ylabel("backgound\nintegrated intensity")

        plt.tight_layout()
        if pdf:
            pdf.savefig(fig)
            plt.close(fig)

    def close(self) -> None:
        self.save()
        self.im.close()  # type: ignore
        self.jm.close()

    @staticmethod
    def get_background(
        h: pandas.DataFrame, im: Imread, transform: Transforms, cnamelist: Sequence[str], r: int = 5, n: int = 4
    ) -> pandas.DataFrame:
        if h.empty:
            h["bg"] = ()
            return h
        else:
            f = []
            for idx, row in transform.inverse.coords_pandas(h, cnamelist).iterrows():
                theta = np.linspace(0, 2 * np.pi, n + 1)[:-1] + np.pi / n
                for i, t in enumerate(theta):
                    nrow = row.copy()
                    nrow["x"] += r * np.cos(t)
                    nrow["y"] += r * np.sin(t)
                    nrow["bg"] = i
                    f.append(nrow)
            f = pandas.concat(f, axis=1).T
            f.index = range(len(f))
            return transform.coords_pandas(
                localisation.tpsuperresseq(
                    f,
                    im,
                    im.sigma,  # type: ignore
                    keep=["x", "y", "s", "e", "theta"],
                    desc="Fitting background localisations",
                ),
                cnamelist,
            )

    @staticmethod
    def show_data(data: pandas.DataFrame, color: Sequence[str], channels: Sequence[int]) -> plt.Figure:
        data = [data.query(f"C == {C}").sort_values("T") for C in channels]
        fig = plt.figure(figsize=(11.69, 8.27))
        gs = GridSpec(2 + len(data), 2, figure=fig, height_ratios=(3,) + (1,) * (1 + len(data)), width_ratios=(2.5, 1))

        t_max = max([d["t"].max() for d in data])

        fig.add_subplot(gs[0, 0])
        for d, c in zip(data, color):
            plt.plot(d["t"], d["i_peak"], c)
        plt.xlim(0, t_max)
        plt.xlabel("time (s)")
        plt.ylabel("peak intensity")

        if len(data) >= 2:
            fig.add_subplot(gs[1, 0])
            g, r = data[-2:]
            plt.plot(
                g["t"],
                np.sqrt(
                    (r["x_um"].to_numpy() - g["x_um"].to_numpy()) ** 2
                    + (r["y_um"].to_numpy() - g["y_um"].to_numpy()) ** 2
                ),
                "k",
            )
            plt.xlim(0, t_max)
            plt.xlabel("time (s)")
            plt.ylabel(rf"{color[-1]}-{color[-2]} distance (μm)")
        for i, (d, c) in enumerate(zip(data, color), 2):
            fig.add_subplot(gs[i, 0])
            plt.plot(d["t"][1:], np.sqrt(np.diff(d["x_um"].to_numpy()) ** 2 + np.diff(d["y_um"].to_numpy()) ** 2), c)
            plt.xlim(0, t_max)
            plt.xlabel("time (s)")
            plt.ylabel(r"step distance (μm)")

        fig.add_subplot(gs[0, 1])
        for d, c in zip(data, color):
            plt.plot(d["x_um"], d["y_um"], c)
        plt.xlabel(r"x (μm)")
        plt.ylabel(r"y (μm)")
        plt.axis("equal")
        plt.gca().invert_yaxis()
        # fig.suptitle(d.name)
        plt.tight_layout()
        return fig

    @staticmethod
    def data_xy(f: pandas.DataFrame | np.ndarray, mode: str = "CellLoc") -> np.ndarray:
        if not isinstance(f, pandas.DataFrame):
            return f
        if mode == "CellLoc":
            if f.empty:
                return np.zeros((0, 1, 6))
            else:
                d = np.zeros((int(f["T"].max() + 1), 1, 6))
                for T in sorted(f["T"].astype(int)):
                    d[T, 0, :] = np.array(f.query(f"T=={T}")[["i_peak", "x", "y", "o", "tiltx", "tilty"]])[0]
                return d
        else:  # DataDigital
            if f.empty:
                return np.zeros((0, 5))
            else:
                d = np.zeros((int(f["T"].max() + 1), 5))
                for T in sorted(f["T"].astype(int)):
                    d[T, :] = np.array(f.query(f"T=={T}")[["x", "y", "i_peak", "T", "link"]])[0]
                for i, j in zip((0, 1, 2, 3), (1, 1, 1 / 2, 0)):
                    d[d[:, -1] == i, -1] = j
                return d

    def save_results(self) -> None:
        path = self.path_out / self.jm.path.stem
        cell = self.cellnr

        for channel, c in zip(self.channel, self.color):
            self.data.query(f"C == {channel}").to_csv(
                f"{path}_cellnr_{cell}_trk_results_{c.lower()}.tsv", sep="\t", index=False
            )
            self.background.query(f"C == {channel}").to_csv(
                f"{path}_cellnr_{cell}_trk_results_bg_{c.lower()}.tsv", sep="\t", index=False
            )
            d = self.data_xy(self.data.query(f"C == {channel}"), "dd")
            np.save(f"{path}dataLocalize3DArray{c}.npy", d)
            np.savetxt(f"{path}_{c.lower()}_digital.txt", d[:, 4], delimiter="\t")

    def write_files(self) -> None:
        print("Write files")
        d = {
            "metadata": self.im.path,
            "max_proj": self.im.path,
            "raw_path": self.im.path,
            "frame_window": [0, self.im.shape["t"]],
        }
        colornames = {
            "r": "red",
            "g": "green",
            "b": "blue",
            "k": "black",
            "w": "white",
            "c": "cyan",
            "y": "yellow",
            "m": "magenta",
        }
        for color in self.color:
            d[f"trk_{color}"] = str(
                self.path_out / f"{self.exp_name}_cellnr_{self.cellnr}_trk_results_{colornames.get(color, color)}.tsv"
            )
        ListFile(d).save(self.path_out / f"{self.exp_name}.list.yml")

    def show_tracking_ncolor(self) -> None:
        """write tif file showing the track"""
        size = 100
        im = self.jm.transpose("tczyx")  # type: ignore
        positions = list(range(len(self.channel)))
        data = [self.data.query(f"C == {C}") for C in self.channel]
        nb_im = im.shape["t"]

        # coordinates of box around yx position of all spots of cell
        box = np.clip(
            np.round(np.nanmean(pandas.concat(data)[["x", "y"]], 0)).astype(int)
            + size // 2 * np.array(((-1, -1), (1, 1))),
            0,
            im.shape["yx"],
        ).flatten()

        channels = self.channel
        square_stamp = [
            np.r_[-5, -5, -5, -5, -5, -5, -5, -5, -4, -3, -2, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, -2, -3, -4],
            np.r_[-5, -4, -3, -2, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, -2, -3, -4, -5, -5, -5, -5, -5, -5, -5],
        ]
        dots_hmm = np.array(((-5, 5), (0, 0)))
        dots_inf = np.array(((0, 0), (-5, 5)))

        def frame_res(jm, loc):
            cropped_im_max = [images.crop(jm[ch].max(0), box[::2], box[1::2]) for ch in channels]

            shape = cropped_im_max[0].shape
            blank = np.zeros(shape)  # empty image

            im_dict = {
                (pos, channel): c_im_m
                for pos, channel, c_im_m in zip(positions, range(1, len(channels) + 1), cropped_im_max)
            }

            for pos, loc in zip(positions, loc):
                sq = np.zeros(shape, "uint16")
                for _, row in loc.iterrows():
                    sq[
                        int(np.round(row["y"] - box[3] + 0.5)) + square_stamp[1],
                        int(np.round(row["x"] - box[2] + 0.5)) + square_stamp[0],
                    ] = 65535
                    if "bin_hmm" in row and row["bin_hmm"] > 0:
                        sq[
                            int(np.round(row["y"] - box[3] + 0.5)) + dots_hmm[1],
                            int(np.round(row["x"] - box[2] + 0.5)) + dots_hmm[0],
                        ] = 65535
                    if "bin_inf" in row and row["bin_inf"] > 0:
                        sq[
                            int(np.round(row["y"] - box[3] + 0.5)) + dots_inf[1],
                            int(np.round(row["x"] - box[2] + 0.5)) + dots_inf[0],
                        ] = 65535

                im_dict[pos, 0] = sq

            if len(channels) > 1:
                qos = max(positions) + 1
                for pos, channel in zip(positions, range(1, len(channels) + 1)):
                    im_dict[qos, channel] = im_dict[pos, channel]
                im_dict[qos, 0] = 65535 * np.any([im_dict[pos, 0] > 0 for pos in positions if (pos, 0) in im_dict], 0)
                frame_positions = positions + [qos]  # prevent mutating positions
            else:
                frame_positions = positions

            # combine cropped image and yx channel
            return [
                np.hstack([im_dict.get((pos, channel), blank) for pos in frame_positions]).astype("uint16")
                for channel in range(len(channels) + 1)
            ]

        with IJTiffFile(
            self.path_out / f"{self.exp_name}_cellnr_{self.cellnr}_track.tif",
            colors=("white",) + tuple(self.color),
            pxsize=im.pxsize_um,
        ) as tif:
            for t in trange(nb_im, desc="Saving frames"):
                frame = frame_res(im[t], [d.query(f"T=={t}") for d in data])
                for c, fr in enumerate(frame):
                    tif.save(fr, c, 0, t)


def main() -> None:
    """imports stuff for running this independently
    rest of execution at the bottom of this file
    """
    from sys import exc_info
    from traceback import format_exception
    from warnings import filterwarnings

    import matplotlib
    from tllab_common.misc import ipy_debug

    matplotlib.use("Agg")
    ipy_debug()
    np.seterr(all="ignore")
    filterwarnings("error", "The unit of the quantity is stripped when")

    """ runs track analysis on all files matching regex in the first argument
        in folder given in the folder argument
    """
    parser = ArgumentParser(description="Display info and save as tif")
    parser.add_argument("files", help="image files", type="str", nargs="*")
    parser.add_argument("-f", "--folder", help="folder with image files", type=Path, default=".")
    parser.add_argument("-L", "--label_channels", help="label channels", nargs="*", type=int)
    parser.add_argument("-P", "--primary_channels", help="primary channels", nargs="*", type=int)
    parser.add_argument("-S", "--secondary_channels", help="secondary channels", nargs="*", type=int)
    parser.add_argument("-m", "--mask_channel", help="mask channel", default=1, type=int)
    parser.add_argument("-d", "--dist_channel", help="search radius in other channel", default=3, type=float)
    parser.add_argument("-e", "--dist_frame", help="search readius in next frame", default=10, type=float)
    parser.add_argument("-p", "--path", help="path out", default=None)
    parser.add_argument("-M", "--mask_method", help="findcells, square or roifile")
    parser.add_argument("-b", "--beadfile", help="beadfile", nargs="*")
    parser.add_argument("-w", "--wavelengths", help="emission wavelengths", nargs="*", type=int)
    parser.add_argument(
        "-c", "--colors", help="colors used when saving data, example: orange lime red", nargs="+", type=str
    )
    args = parser.parse_args()

    fnames = sum([args.folder.glob(file) for file in args.files], [])
    err = []
    if len(fnames) == 1:
        with TrackAnalysis(
            fnames[0],
            args.label_channels,
            args.primary_channels,
            args.secondary_channels,
            args.mask_channel,
            args.dist_channel,
            args.dist_frame,
            args.path,
            (args.mask_method,),
            args.beadfile,
            args.wavelengths,
            args.colors,
        ) as track:
            track.save_tracking()
            track.plot()
    else:
        for fname in fnames:
            try:
                cprint(f"<Working on: {fname}:g.b>")
                with TrackAnalysis(
                    fname,
                    args.label_channels,
                    args.primary_channels,
                    args.secondary_channels,
                    args.mask_channel,
                    args.dist_channel,
                    args.dist_frame,
                    args.path,
                    (args.mask_method,),
                    args.beadfile,
                    args.wavelengths,
                    args.colors,
                ) as track:
                    track.save_tracking()
                    track.plot()
            except Exception:  # noqa
                err.append((fname, format_exception(*exc_info())))  # noqa
        if err:
            print("Occurred errors:")
            pprint(err)
            ename = Path(args.folder.replace("data", "analysis")) / "track_analysis_errors.pk"
            with open(ename, "wb") as f:
                pickle.dump(err, f)  # type: ignore


if __name__ == "__main__":
    main()
