"""Independent marker outlines for recorded QC statuses on working measurements."""
from math import sqrt
from hashlib import sha256
import json


def has_qc_status(status):
    return isinstance(status, str) and status.strip().casefold() not in ("", "unchecked")


def draw_qc_outlines(ax, project, artists, column_ids):
    snapshot = project.view.get("plot_measurement_qc")

    def statuses(rid):
        values = []
        for cid in column_ids:
            if snapshot is not None:
                item = snapshot.get(cid, {}).get(rid)
                status = item[2] if item is not None else None
            else:
                measurement = project.science.current(rid, cid)
                status = measurement.qc_status if measurement is not None else None
            if has_qc_status(status):
                values.append(status.strip())
        return tuple(sorted(set(values)))

    groups = {}
    for artist in artists:
        if artist.axes is not ax:
            continue
        diameters = artist.get_sizes()
        for index, (rid, point) in enumerate(zip(artist._autopyro_row_ids, artist.get_offsets(), strict=True)):
            key = statuses(rid)
            if key:
                x, y, sizes = groups.setdefault(key, ([], [], []))
                x.append(point[0])
                y.append(point[1])
                sizes.append((sqrt(diameters[index if len(diameters) > 1 else 0]) + 4) ** 2)
    outlines = []
    markers = ("o", "s", "D", "^", "v", "p", "h")
    for index, (key, (x, y, sizes)) in enumerate(sorted(groups.items())):
        digest = sha256(json.dumps(key, ensure_ascii=False).encode("utf-8")).digest()
        marker = markers[int.from_bytes(digest[:4], "big") % len(markers)]
        outlines.append(ax.scatter(
            x, y, s=sizes, marker=marker, facecolors="none",
            edgecolors="#b36b00", linewidths=1.4, zorder=5, label="QC · " + " / ".join(key),
            gid="measurement-qc-outline" if index == 0 else f"measurement-qc-outline-{index + 1}"))
    return outlines
