import json
import os
import re
from collections import defaultdict
from typing import Any, Self, Sequence, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from affine import Affine
from rasterio.plot import reshape_as_image
from scipy import ndimage as ndi
from shapely import contains, simplify
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import linemerge, polygonize, unary_union
from skimage import color, exposure, filters, io, measure, morphology, transform
from sklearn.cluster import KMeans
from svgelements import SVG, Circle, Ellipse, Path, Rect

# from Real-ESRGAN import inference_realesrgan


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
        border_points = []
        for element in self.svg.elements(lambda elem: isinstance(elem, Rect)):
            # if isinstance(element, Circle):
            #     border_points.append(element.point(0))
            border_box = element.bbox()
            border_points.extend(
                [
                    (border_box[0], border_box[1]),
                    (border_box[2], border_box[3]),
                ]
            )

        edge_points = np.array(border_points)
        max_point = np.abs(edge_points.min(axis=0))
        min_point = np.abs(edge_points.max(axis=0))

        return min_point, max_point  # (x_min, y_min), (x_max, y_max)

    def _extrapolate_curve_ends(
        self, line: LineString, length_ratio: float = 0.1
    ) -> LineString:
        distance = line.length * length_ratio
        p1, p2 = line.coords[:2]  # first_segment
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        start_point = Point(
            p1[0] - distance * np.cos(angle), p1[1] - distance * np.sin(angle)
        )

        pn1, pn = line.coords[-2:]  # last_segment
        angle = np.arctan2(pn[1] - pn1[1], pn[0] - pn1[0])
        end_point = Point(
            pn[0] + distance * np.cos(angle), pn[1] + distance * np.sin(angle)
        )

        return LineString([*start_point.coords, *line.coords[1:-1], *end_point.coords])

    def _generate_curves(
        self, step: int, divider: bool = False, log: bool = False
    ) -> dict:
        curves_dict = {}
        for element in self.svg.elements(lambda elem: isinstance(elem, Path)):
            points = element.npoint(np.linspace(0, 1, step))
            points[:, 0] = np.abs(points[:, 0])
            points[:, 1] += self.image_coords[:, 1].sum()  # = points[:, 1] +
            # y_min + y_max
            points = self._convert_coords(points, log=log)

            if found_label := re.findall(self.LABEL_TAG, element.string_xml()):
                label = found_label[0]
                label_name, *label_values = re.search(self.LABEL_PARSE, label).groups()
                curves_dict[label] = {
                    "color": element.stroke.hex,
                    "width": element.stroke_width,
                    "label": {
                        "name": label_name,
                        "value": [val for val in label_values if val],
                        "divider": divider,
                    },
                    "equation": {"curve_type": None, "params": []},
                    "points": {
                        "x": points[:, 0].tolist(),
                        "y": points[:, 1].tolist(),
                    },
                }

        return curves_dict

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

            if found_label := re.findall(self.LABEL_TAG, element.string_xml()):
                label_name, *label_values = re.search(
                    self.LABEL_PARSE, found_label[0]
                ).groups()
                markers_list.append((Point(point), (label_name, label_values[0])))

        return markers_list

    def _generate_areas(
        self, curves_dict: dict, markers_list: list, length_ratio: float = 0.1
    ) -> dict:
        lines = [
            self._extrapolate_curve_ends(
                simplify(
                    LineString(list(zip(*value["points"].values()))), tolerance=0.05
                ),
                length_ratio=length_ratio,
            )
            for value in curves_dict.values()
        ]
        # labels = [value["label"] for value in curves_dict.values()]

        bounds = Polygon(
            [
                (self.plot_coords[0, 0], self.plot_coords[0, 1]),
                (self.plot_coords[0, 0], self.plot_coords[1, 1]),
                (self.plot_coords[1, 0], self.plot_coords[1, 1]),
                (self.plot_coords[1, 0], self.plot_coords[0, 1]),
            ]
        )

        lines.append(bounds.boundary)
        lines = linemerge(lines)
        lines = unary_union(lines)
        polygons = [
            polygon for polygon in polygonize(lines) if polygon.area > 1
        ]  # if not np.allclose(polygon.area, 0)

        markers_geometry = [point[0] for point in markers_list]
        markers_labels = [point[1] for point in markers_list]
        inclusions = np.array([contains(poly, markers_geometry) for poly in polygons])

        areas_dict = {}
        for i, j in np.argwhere(inclusions):
            x, y = polygons[i].exterior.coords.xy
            name, values = markers_labels[j]
            areas_dict[f"{name}: {values}"] = {
                "label": {"name": name, "value": values},
                "points": {
                    "x": x.tolist(),
                    "y": y.tolist(),
                },
            }

        # areas_dict = {}
        # for label, (i, line) in zip(labels, enumerate(lines)):
        #     sliced_part, bounds = split(bounds, line).geoms
        #     print(1)
        #     name = label["name"]
        #     values = label["value"]
        #     if i != len(lines) - 1:
        #         x, y = sliced_part.exterior.coords.xy
        #         areas_dict[f"{name}: {values[0]}"] = {
        #             "label": {"name": name, "value": values[0]},
        #             "points": {
        #                 "x": x.tolist(),
        #                 "y": y.tolist(),
        #             },
        #         }
        #     else:
        #         x, y = sliced_part.exterior.coords.xy
        #         areas_dict[f"{name}: {values[0]}"] = {
        #             "label": {"name": name, "value": values[0]},
        #             "points": {
        #                 "x": x.tolist(),
        #                 "y": y.tolist(),
        #             },
        #         }
        #         x, y = bounds.exterior.coords.xy
        #         areas_dict[f"{name}: {values[1]}"] = {
        #             "label": {"name": name, "value": values[1]},
        #             "points": {
        #                 "x": x.tolist(),
        #                 "y": y.tolist(),
        #             },
        #         }

        return areas_dict

    def to_dict(
        self,
        step: int = 1000,
        divider: bool = False,
        log: bool = False,
        grid: bool = True,
        legend: bool = True,
    ) -> dict[str, Any]:
        curves = self._generate_curves(step=step, divider=divider, log=log)
        areas = (
            self._generate_areas(curves, self._find_area_markers(log=log))
            if divider
            else {}
        )

        possible_labels = defaultdict(list)
        for curve in curves.values():
            lab = curve["label"]
            possible_labels[lab["name"]].extend(lab["value"])

        chart_dict = {
            "name": self.name,
            "title": self.title,
            "settings": {
                "xlim": self.plot_coords[:, 0].tolist(),  # [x_min_real, x_max_real]
                "ylim": self.plot_coords[:, 1].tolist(),  # [y_min_real, y_max_real]
                "log": log,
                "grid": grid,
                "legend": legend,
            },
            "labels": dict(possible_labels),
            "data": {"curves": curves, "areas": areas},
        }

        return chart_dict

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


# class ImagePlotDigitizer:
#     # __slots__ = "image", ""

#     def __init__(
#         self,
#         image: np.ndarray,
#         binary_image: np.ndarray,
#         lines: list[np.ndarray],
#         curves: list[np.ndarray],
#         transform_points: list[np.ndarray],
#     ) -> None:
#         self.image = image
#         self.binary_image = binary_image
#         self.lines = lines
#         self.curves = curves
#         self.transform_points = transform_points

#     @classmethod
#     def from_file(cls, image_file: str | os.PathLike):
#         image = io.imread(image_file)
#         if image.shape[2] > 3:
#             image = color.rgba2rgb(image)

#         return cls(image)

#     def from_svg(cls, svg_file: str | os.PathLike:
#         return SVGParse(svg_file).svg2json()

#     def _find_lines(self, angles):
#         h, theta, d = transform.hough_line(self.binary_image, theta=angles)

#         for i, (_, angle, dist) in enumerate(
#             zip(*transform.hough_line_peaks(h, theta, d))
#         ):
#             (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
#             ang = np.tan(angle + np.pi / 2)
#             if np.isclose(ang, 0, atol=0.01):
#                 ang = 0
#             elif np.isinf(ang):
#                 pass

#         return (x1, y1), (x2, y2)

#     def create_binary(self, block_size: int = 10, offset: int = 10):
#         gray_image = color.rgb2gray(self.image)
#         local_thresh = filters.threshold_local(gray_image, block_size, offset=offset)
#         self.binary_image = gray_image < local_thresh

#     def find_axis_lines(self):
#         x_angles = np.linspace(3 * np.pi / 8, 5 * np.pi / 8, 90, endpoint=True)
#         y_angles = np.linspace(-np.pi / 8, np.pi / 8, 90, endpoint=False)
#         other_angles = np.linspace(np.pi / 8, 3 * np.pi / 8, 180, endpoint=False)

#         x_axis = self._find_lines(x_angles)
#         y_axis = self._find_lines(y_angles)

#     def find_curves(self):
#         # contours = measure.find_contours(self.binary_image, 0.8)
#         skeleton = morphology.skeletonize(self.binary_image)

#         for contour in skeleton:
#             pass
#             # segmentation.flood_fill(binary_inv, (contour[0][1]-1, contour[0][0]-1), 255)
#             # contour[:, 1], contour[:, 0]  x, y
