from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas
from inflection import underscore
from ndbioimage import Imread
from ruamel import yaml
from tllab_common.misc import Struct
from tqdm.auto import tqdm


class ListFile(list):
    def __init__(self, *entries):
        super().__init__()
        self.file = None
        self.info = {}
        if entries is not None:
            self.extend(entries)

    def sort(self, key=None, reverse=False):
        if key is None:
            key = ["".join([i[k] for k in ("raw_path", "trk_r", "trk_g")]) for i in self]
        idx = [i[0] for i in sorted(enumerate(key), key=lambda x: x[1], reverse=reverse)]
        tmp = self.copy()
        for i, j in enumerate(idx):
            self[i] = tmp[j]
        del tmp

    @property
    def on(self):
        """Return a new listFile with only the entries not commented out with #'s."""
        return ListFile(*[i for i in self if i.get("use", True)])

    @property
    def off(self):
        """Return a new listFile with only the entries commented out with #'s."""
        return ListFile(*[i for i in self if not i.get("use", True)])

    @classmethod
    def load(cls, file):
        file = Path(file).resolve()
        with open(file, "r") as f:
            content = f.read()
        new = cls()
        new.file = file

        if file.suffix == ".yml":
            # pseudo yml parsing mostly ignoring #
            sub_pattern = re.compile(r"^(\s*#?\s*)+")
            use_pattern = re.compile(r"(:?(:?^|\n)\s*#|^\s*$)")
            for item in re.split(r"(:?^|\n)[\s#]*-", content):
                comment, body, use = "", {}, True
                for line in item.splitlines():
                    y = yaml.YAML(typ="safe").load(sub_pattern.sub("", line))
                    if isinstance(y, dict):
                        body.update(**y)
                        use = use and not use_pattern.match(line)
                    else:
                        comment += sub_pattern.sub("", line)
                if body:
                    d = Struct(use=use, comment=comment, **body)
                    new.append({underscore(key): value for key, value in d.items()})
        else:
            list_files = re.findall(r"\[(.*)]", content, re.DOTALL)  # everything between listFiles=[ ... ]
            if len(list_files):
                list_files = list_files[0][1:-1]
                items = re.findall(r"#?[^{}]*{[^{}]*}", list_files, re.DOTALL)  # N x (comment + entry)

                for item in items:
                    entry = re.findall(r"#?\s*{[^{}]*}", item, re.DOTALL)[0]
                    comment = re.findall(r"^[^{}]*", item, re.DOTALL)[0]
                    comment = re.sub(r"^,?[\r\n\s#]*", "", comment)
                    comment = re.sub(r"[\r\n\s#]*$", "", comment)
                    use = not entry.startswith("#")
                    entry = re.sub(r"^(\s*#)+", "", entry, flags=re.MULTILINE)
                    entry = re.sub(r"\t", " ", entry, flags=re.MULTILINE)
                    d = Struct(use=use, comment=comment)
                    d.update(yaml.YAML(typ="safe").load(entry))
                    new.append({underscore(key): value for key, value in d.items()})
        return new

    def get_info(self, n):
        if not isinstance(n, str):
            n = self[n]["raw_path"]
        if n not in self.info:
            with Imread(n) as im:
                self.info[n] = im.summary
        return self.info[n]

    def __repr__(self):
        """This is also exactly how the file will be saved."""
        if self.file is not None and self.file.suffix == ".yml":
            s = ""
            for item in self:
                use = item.get("use", True)
                s += "-\n" if use else "# -\n"
                if "comment" in item and item["comment"] is not None:
                    if len(item["comment"]):
                        s += f"# {item['comment']}\n"

                for k, v in item.items():
                    if k not in ("use", "comment"):
                        s += "  " if use else "#   "
                        s += f"{k}: {v}\n"
                s += "\n"
            return s
        else:  # py
            s = "# coding: utf-8\nlistFiles=["
            for item in self:
                s += "\n    "
                if "comment" in item:
                    if len(item["comment"]):
                        s += "# {}\n    ".format(item["comment"])
                if not item.get("use", True):
                    s += "# "
                s += "{"
                for i, (k, v) in enumerate(item.items()):
                    if k not in ("use", "comment"):
                        s += "'{}': {},\n    ".format(k, v.__repr__())
                        if not item.get("use", True):
                            s += "# "
                s += "},\n"
            s += "]"
            return s

    def save(self, file=None):
        """This just wraps __repr__ and saves its output to a file."""
        file = Path(file or self.file).resolve()
        self.file = file
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w") as f:
            print(self, file=f)  # type: ignore

    def copy(self):
        return deepcopy(self)

    def __eq__(self, other):
        if len(self) == 0 and len(other) == 0:
            return True
        elif len(self) != len(other):
            return False
        else:
            for i, j in zip(self, other):
                for key, value in i.items():
                    if key not in j:
                        return False
                    if j[key] != value:
                        return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return ListFile(*super().__getitem__(item))
        elif isinstance(item, (list, tuple)):
            return ListFile(*[super().__getitem__(i) for i in item])
        else:
            return super().__getitem__(item)


class ExpData(list):
    # TODO: deprecate this
    def __init__(self, lf=None, ignore_comments=False):
        if lf is None:
            super().__init__()
            return
        if isinstance(lf, list):
            super().__init__(lf)
            return
        super().__init__()
        if not isinstance(lf, ListFile):
            lf = Path(lf).expanduser()
            lf = ListFile.load(lf)
        if not ignore_comments:
            lf = lf.on

        for a in tqdm(lf, desc="Loading experimental data"):
            d = Struct()
            colors = tuple([key[4:] for key in a.keys() if key.startswith("trk_")])
            d["colors"] = colors
            for c in colors:
                if a[f"trk_{c}"].endswith(".tsv"):
                    d[f"trk_{c}"] = pandas.read_csv(a[f"trk_{c}"], sep="\t")[
                        ["x", "y", "i_peak", "T", "link"]
                    ].to_numpy()
                else:
                    d[f"trk_{c}"] = np.loadtxt(a[f"trk_{c}"])

            if "frame_window" in a:
                d.frame_window = a["frame_window"]
            elif "frameWindow" in a:
                d.frame_window = a["frameWindow"]

            if "raw_path" in a:
                d.raw_path = a["raw_path"]
            elif "rawPath" in a:
                d.raw_path = a["rawPath"]

            try:
                d.name = re.findall(r"^(.*)_[^_]+\.(?:txt|tsv)$", a[f"trk_{colors[0]}"])[0]
            except IndexError:
                d.name = a[f"trk_{colors[0]}"]

            if "max_proj" in a:
                d.max_proj = a["max_proj"]
            elif "maxProj" in a:
                d.max_proj = a["maxProj"]

            if "time_interval" in a:
                d.dt = float(a["time_interval"])
            else:
                with Imread(d.raw_path) as im:
                    d.dt = im.timeinterval

            i = 1 if a[f"trk_{colors[0]}"].endswith("trk") else 2  # orbital vs widefield files
            d.t = d[f"trk_{colors[0]}"][:, -i] * d.dt

            for c in colors:
                d[c] = d[f"trk_{c}"][:, -(i + 1)]

            if "frame_window" not in d:
                d.frame_window = [0, d.t.shape[0]]
            else:
                if d.frame_window[0] < 0:
                    d.frame_window[0] = 0
                if d.frame_window[1] > d.t.shape[0] - 1:
                    d.frame_window[1] = d.t.shape[0] - 1

            self.append(d)

    def __getitem__(self, item):
        r = super().__getitem__(item)
        return self.__class__(r) if isinstance(r, list) else r
