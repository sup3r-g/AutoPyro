import json
import os
import re
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Self, Union

import numpy as np

# from shapely import contains, simplify
from shapely.geometry import LineString, Point, Polygon

# from shapely.ops import linemerge, polygonize, unary_union
from svgelements import SVG, Circle, Ellipse, Path, Rect, Image

from AutoPyro.core.json_models import (
    AreaModel,
    CurveModel,
    DataModel,
    EquationModel,
    LabelModel,
    PlotModel,
    PlotSettingsModel,
    PointModel,
    StyleModel,
)


def extrapolate_curve(line: LineString, length_ratio: float = 0.1) -> LineString:
    distance = line.length * length_ratio

    # First Segment (1, 2)
    point_1, point_2 = line.coords[:2]
    angle = np.arctan2(point_2[1] - point_1[1], point_2[0] - point_1[0])
    start_point = Point(
        point_1[0] - distance * np.cos(angle), point_1[1] - distance * np.sin(angle)
    )

    # Last Segment (n-1, n)
    point_n1, point_n = line.coords[-2:]
    angle = np.arctan2(point_n[1] - point_n1[1], point_n[0] - point_n1[0])
    end_point = Point(
        point_n[0] + distance * np.cos(angle), point_n[1] + distance * np.sin(angle)
    )

    return LineString([*start_point.coords, *line.coords[1:-1], *end_point.coords])


class SVGParse:
    __slots__ = "svg", "name", "title", "image_coords", "plot_coords"
    LABEL_TAG = re.compile(r'label="(.+?)"')
    LABEL_PARSE = (
        r"([\w\W ()]+): \(?([-/\d \w\.%]+)\)?(?:, ?)?(?:\(?([-/\d \w\.%]+)\)?)?"
    )

    def __init__(
        self,
        file_path: str | os.PathLike,
        plot_coords: list[list[float]],
        image_coords: Union[list[list[float]], None] = None,
    ) -> None:
        file_path = Path(file_path)
        self.svg = SVG.parse(file_path, transform="rotate(180)")
        self.name = file_path.stem
        # os.path.splitext(os.path.basename(file_path))[0]
        self.title = file_path.parent.name
        # os.path.basename(os.path.dirname(file_path))
        # (x_min, y_min), (x_max, y_max)
        self.image_coords = np.asarray(
            image_coords if image_coords else self._find_image_coords()
        )
        # (x_min_real, y_min_real), (x_max_real, y_max_real)
        self.plot_coords = np.asarray(plot_coords)

    @classmethod
    def from_svg(
        cls, svg_file: str | os.PathLike, plot_coords: list[list[float]]
    ) -> Self:
        ext = Path(svg_file).suffix
        if ext != ".svg":
            raise ValueError("Supplied file not in SVG format")

        return cls(svg_file, plot_coords)

    def _label_from_element(self, element):
        # TODO: rework label logic because we won't use the old one
        if found_label := re.findall(self.LABEL_TAG, element.string_xml()):
            label = found_label[0]
            label_name, *label_values = re.search(self.LABEL_PARSE, label).groups()

            return label, label_name, label_values

        return None, None, None

    def _convert_coords(self, coords: list[tuple[float]], log: bool = False):
        if log:
            return 10 ** (
                (
                    (coords - self.image_coords[0])  # min
                    * (
                        np.log10(self.plot_coords[1]) - np.log10(self.plot_coords[0])
                    )  # max - min
                    / (self.image_coords[1] - self.image_coords[0])  # max - min
                )
                + np.log10(self.plot_coords[0])
            )  # min

        return (
            (coords - self.image_coords[0])  # min
            * (self.plot_coords[1] - self.plot_coords[0])  # max - min
            / (self.image_coords[1] - self.image_coords[0])  # max - min
        ) + self.plot_coords[
            0
        ]  # min

    def _find_image_coords(self) -> tuple[float, ...]:
        # border_points = self.svg.union_bbox(
        #     self.svg.elements(lambda elem: not isinstance(elem, Image)),
        # )
        # edge_points.reshape(2, -1)

        # Get all Rectangles present in the image
        border_points = []
        for element in self.svg.elements(lambda elem: isinstance(elem, Rect)):
            border_box = element.bbox()
            border_points.extend(
                [
                    (border_box[0], border_box[1]),
                    (border_box[2], border_box[3]),
                ]
            )

        # Find edge points
        edge_points = np.array(border_points)
        max_point = np.abs(edge_points.min(axis=0))
        min_point = np.abs(edge_points.max(axis=0))

        return min_point, max_point  # (x_min, y_min), (x_max, y_max)

    def _curves(self, step: int, divider: bool = False, log: bool = False) -> dict:
        curves_dict = {}
        for element in self.svg.elements(lambda elem: isinstance(elem, Path)):
            points = element.npoint(np.linspace(0, 1, step))
            points[:, 0] = np.abs(points[:, 0])
            points[:, 1] += self.image_coords[:, 1].sum()  # = points[:, 1] +
            # y_min + y_max
            points = self._convert_coords(points, log=log)

            label, label_name, label_values = self._label_from_element(element)
            if label:
                curves_dict[label] = asdict(
                    CurveModel(
                        style=StyleModel(
                            color=element.stroke.hex,
                            width=element.stroke_width,
                        ),
                        label=LabelModel(
                            name=label_name,
                            value=[val for val in label_values if val],
                        ),
                        divider=divider,
                        equation=EquationModel(curve_type=None, params=[]),
                        points=[
                            PointModel(
                                x=points[:, 0].tolist(),
                                y=points[:, 1].tolist(),
                                label=LabelModel("", ""),
                            )
                        ],
                    )
                )

        return curves_dict

    # TODO: Remove this method because we won't need it in with attribute labels
    # `autopyro:label=value``
    def _find_area_markers(self, log: bool = False) -> list:
        markers_list = []
        for element in self.svg.elements(
            lambda elem: isinstance(elem, (Circle, Ellipse))
        ):
            point = np.atleast_2d(element.point(0))
            point[:, 0] = np.abs(point[:, 0])
            point[:, 1] += self.image_coords[:, 1].sum()  # = point[:, 1] +
            # y_min + y_max
            point = self._convert_coords(point, log=log)

            _, label_name, label_values = self._label_from_element(point)
            if label_name:
                markers_list.append((Point(point), (label_name, label_values[0])))

        return markers_list

    # def _areas(
    #     self, curves_dict: dict, markers_list: list, length_ratio: float = 0.1
    # ) -> dict:
    #     lines = [
    #         extrapolate_curve(
    #             simplify(
    #                 LineString(list(zip(*value["points"].values()))), tolerance=0.05
    #             ),
    #             length_ratio=length_ratio,
    #         )
    #         for value in curves_dict.values()
    #     ]
    #     # labels = [value["label"] for value in curves_dict.values()]

    #     bounds = Polygon(
    #         [
    #             (self.plot_coords[0, 0], self.plot_coords[0, 1]),
    #             (self.plot_coords[0, 0], self.plot_coords[1, 1]),
    #             (self.plot_coords[1, 0], self.plot_coords[1, 1]),
    #             (self.plot_coords[1, 0], self.plot_coords[0, 1]),
    #         ]
    #     )

    #     lines.append(bounds.boundary)
    #     lines = unary_union(linemerge(lines))
    #     polygons = [polygon for polygon in polygonize(lines) if polygon.area > 1]

    #     # TODO: remove, since we won't need this
    #     markers_geometry = [point[0] for point in markers_list]
    #     markers_labels = [point[1] for point in markers_list]
    #     inclusions = np.array([contains(poly, markers_geometry) for poly in polygons])

    #     areas_dict = {}
    #     for i, j in np.argwhere(inclusions):
    #         x, y = polygons[i].exterior.coords.xy
    #         name, values = markers_labels[j]
    #         areas_dict[f"{name}: {values}"] = asdict(
    #             AreaModel(
    #                 label=LabelModel(name=name, value=values),
    #                 points=[
    #                     PointModel(x=x.tolist(), y=y.tolist(), label=LabelModel("", ""))
    #                 ],
    #             )
    #         )

    #     return areas_dict

    def _areas(self, step: int, log: bool = False) -> dict:
        areas_dict = {}
        for element in self.svg.elements(lambda elem: isinstance(elem, Path)):
            points = element.npoint(np.linspace(0, 1, step))
            points[:, 0] = np.abs(points[:, 0])
            points[:, 1] += self.image_coords[:, 1].sum()  # = points[:, 1] +
            # y_min + y_max
            points = self._convert_coords(points, log=log)

            label, label_name, label_values = self._label_from_element(element)
            if label:
                areas_dict[label] = asdict(
                    AreaModel(
                        label=LabelModel(name=label_name, value=label_values),
                        points=[
                            PointModel(
                                x=points[:, 0].tolist(),
                                y=points[:, 1].tolist(),
                                label=LabelModel("", ""),
                            )
                        ],
                    )
                )

        return areas_dict

    def to_dict(
        self,
        step: int = 1000,
        divider: bool = False,
        log: bool = False,
        grid: bool = True,
        legend: bool = True,
    ) -> dict[str, Any]:
        curves = self._curves(step=step, divider=divider, log=log)
        areas = self._areas(curves, self._find_area_markers(log=log)) if divider else {}

        # Ugly, very ugly
        possible_labels = defaultdict(list)
        for curve in curves.values():
            lab = curve["label"]
            possible_labels[lab["name"]].extend(lab["value"])

        chart_dict = PlotModel(
            name=self.name,
            title=self.title,
            settings=PlotSettingsModel(
                xlim=self.plot_coords[:, 0].tolist(),  # [x_min_real, x_max_real]
                ylim=self.plot_coords[:, 1].tolist(),  # [y_min_real, y_max_real]
                log=log,
                grid=grid,
                legend=legend,
            ),
            labels=dict(possible_labels),
            data=DataModel(curves=curves, areas=areas, points=[]),
        )

        return asdict(chart_dict)

    def to_json(
        self,
        step: int = 1000,
        divider: bool = False,
        log: bool = False,
        grid: bool = True,
        legend: bool = True,
    ) -> None:
        with open(f"{self.name}.json", "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(step, divider, log, grid, legend), fp=fp, indent=4)
