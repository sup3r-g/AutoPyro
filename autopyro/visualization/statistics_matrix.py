"""Shared rendering for statistical matrix previews and image exports."""
from textwrap import fill


MATRIX_STYLE_FIELDS = ("background", "font_family", "title_size")
SCATTER_STYLE_FIELDS = ("color", "alpha")


def matrix_style(spec):
    from matplotlib.colors import is_color_like

    style = spec.get("style", {})
    if not isinstance(style, dict):
        raise ValueError("Неизвестный формат шаблона")
    for key in ("background", "color"):
        if key in style and (not isinstance(style[key], str) or not is_color_like(style[key])):
            raise ValueError("Неизвестный формат шаблона")
    if ("font_family" in style
            and (not isinstance(style["font_family"], str) or not style["font_family"].strip())):
        raise ValueError("Неизвестный формат шаблона")
    for key, lower, upper in (("title_size", 1, 100), ("alpha", 0, 1)):
        if key in style and (type(style[key]) not in (int, float) or not lower <= style[key] <= upper):
            raise ValueError("Неизвестный формат шаблона")
    return style


def apply_matrix_style(figure, title, style):
    if "background" in style:
        figure.set_facecolor(style["background"])
    if "title_size" in style:
        title.set_fontsize(style["title_size"])
    if "font_family" in style:
        from matplotlib.text import Text

        for text in figure.findobj(Text):
            text.set_fontfamily(style["font_family"])


def correlation_matrix_figure(matrix_spec, label):
    from matplotlib.figure import Figure
    import numpy as np

    style = matrix_style(matrix_spec)
    method = matrix_spec["method"]
    column_ids = [column["id"] for column in matrix_spec["columns"]]
    size = len(column_ids)
    entries = matrix_spec["pairs"]
    positions = {cid: position for position, cid in enumerate(column_ids)}
    names = [fill(column["name"], width=24) for column in matrix_spec["columns"]]
    matrix = np.full((size, size), np.nan)
    extent = 160 + 62 * size
    figure = Figure(figsize=(max(440, extent + 100) / 100, max(320, extent) / 100),
                    dpi=100, layout="constrained")
    axes = figure.add_subplot()
    for entry in entries:
        x, y = positions[entry["x_id"]], positions[entry["y_id"]]
        coefficient = entry[method]
        matrix[x, y] = matrix[y, x] = coefficient if coefficient is not None else np.nan
        value = f"{coefficient:.2f}" if coefficient is not None else "—"
        color = "white" if coefficient is not None and abs(coefficient) >= 0.72 else "#202020"
        for row, col in ((x, y), (y, x)):
            axes.text(col, row, f"{value}\nN={entry['n_pairs']}",
                      ha="center", va="center", color=color, fontsize=8)
    axes.set_facecolor("#e5e7eb")
    heatmap = axes.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
    axes.set_xticks(range(size), names, rotation=45, ha="right")
    axes.set_yticks(range(size), names)
    title = axes.set_title(fill(f"{label} · {matrix_spec['group_label']}", width=64))
    figure.colorbar(heatmap, ax=axes, label=label)
    apply_matrix_style(figure, title, style)
    return figure


def scatter_matrix_figure(matrix_spec, values, label):
    from matplotlib.figure import Figure
    import numpy as np

    style = matrix_style(matrix_spec)
    names = [fill(f"{c['name']} [{c['unit']}]" if c["unit"] else c["name"], width=22)
             for c in matrix_spec["columns"]]
    size = len(names)
    extent = 100 + 145 * size
    figure = Figure(figsize=((extent + 80) / 100, extent / 100), dpi=100, layout="constrained")
    axes = figure.subplots(size, size, sharex="col", squeeze=False)
    for row in range(size):
        for col in range(size):
            panel = axes[row, col]
            if col > row:
                panel.set_visible(False)
                continue
            if "background" in style:
                panel.set_facecolor(style["background"])
            x = values[col]
            if row == col:
                present = np.isfinite(x)
                if present.any():
                    panel.hist(x[present], bins=12, color=style.get("color"))
            else:
                y = values[row]
                present = np.isfinite(x) & np.isfinite(y)
                if col > 0:
                    panel.sharey(axes[row, 0])
                if present.any():
                    panel.scatter(x[present], y[present], s=9, color=style.get("color"),
                                  alpha=style.get("alpha", 0.5))
            panel.text(0.04, 0.96, f"N={present.sum()}", transform=panel.transAxes, va="top")
            panel.tick_params(labelsize=8, labelleft=col == 0 or row == col)
            if col == 0 and row > 0:
                panel.set_ylabel(names[row])
            if row == size - 1:
                panel.set_xlabel(names[col])
    title = figure.suptitle(fill(f"{label} · {matrix_spec['group_label']}", width=64))
    apply_matrix_style(figure, title, style)
    return figure
