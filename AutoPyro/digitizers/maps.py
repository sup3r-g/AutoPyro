import os
from typing import Self, Sequence, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from affine import Affine
from rasterio.plot import reshape_as_image
from scipy import ndimage as ndi
from shapely.geometry import MultiPolygon, Polygon
from skimage import color, exposure, filters, io, measure, morphology, transform
from sklearn.cluster import KMeans

# from Real-ESRGAN import inference_realesrgan


class MapDigitizer:
    __slots__ = "image", "elements", "crs", "transform", "scale"

    def __init__(self, image: np.ndarray, crs: str, transform: Affine) -> None:
        self.image = image
        self.crs = crs
        self.transform = transform
        self.elements = []
        self.scale = 1.0

    @classmethod
    def from_file(
        cls, image_file: Union[str, os.PathLike], crs: str, transform: Affine
    ) -> Self:
        image = io.imread(image_file)

        return cls(image=image, crs=crs, transform=transform)

    @classmethod
    def from_tif(
        cls,
        tif_file: Union[str, os.PathLike],
        upscale: bool = False,
        downscale_by: float = 1.0,
    ) -> Self:
        with rio.open(tif_file) as src:
            image = reshape_as_image(src.read().astype(rio.uint8))
            crs = src.crs
            transform = src.transform

        return cls(
            image=cls._preprocess_image(
                image, upscale=upscale, downscale_by=downscale_by
            ),
            crs=crs,
            transform=transform,
        )

    def _preprocess_image(
        self, image: np.array, upscale: bool = False, downscale_by: float = 1.0
    ) -> np.array:
        if upscale:
            downscale_by = 2.0 if downscale_by is None else max(2.0, 1 / downscale_by)

        if downscale_by:
            image = transform.rescale(
                image, 1 / downscale_by, anti_aliasing=True, channel_axis=-1
            )
            self.scale = downscale_by

        image = exposure.equalize_adapthist(image, kernel_size=8)

        return image

    def _fix_image() -> None:
        pass

    def separate_color(
        self, n_clusters: int, plot_pie_chart: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        image_lab = color.rgb2lab(self.image).reshape(
            (self.image.shape[1] * self.image.shape[0], self.image.shape[2])
        )
        clustering_algorithm = KMeans(n_clusters=n_clusters).fit(image_lab)
        centroid_image = clustering_algorithm.cluster_centers_
        labels_image = clustering_algorithm.labels_

        labels, percent = np.unique(labels_image, return_counts=True)
        percent /= labels_image.shape[0]
        colors = color.lab2rgb(centroid_image)

        if plot_pie_chart:
            plt.pie(percent, colors=colors, labels=labels)

        labels_array = labels_image.reshape(self.image.shape[0], self.image.shape[1])

        return labels_array, colors

    def create_polygons(
        self, chosen_labels: Sequence, min_size: int = 100, kernel: np.array = 5
    ):
        labels_array = self.separate_color()

        self.elements.clear()
        for label in chosen_labels:
            label_bool = labels_array == label
            label_bool = np.fliplr(np.rot90(label_bool, k=3))

            binary = ndi.binary_fill_holes(label_bool, structure=np.ones((20, 10)))

            binary = filters.median(label_bool, morphology.square(kernel))
            binary = morphology.remove_small_objects(binary, min_size)
            binary = morphology.remove_small_holes(binary, min_size)
            binary = morphology.binary_closing(binary, morphology.disk(kernel))

            binary = ndi.binary_fill_holes(binary)

            contours = measure.find_contours(binary, 0.9)
            self.elements.append(
                MultiPolygon([Polygon(contour) for contour in contours])
            )

        final_transform = np.array(self.transform.to_shapely()) * np.array(
            [self.scale, 1, 1, self.scale, 1, 1]
        )
        gdf = gpd.GeoSeries(self.elements, crs=self.crs)
        gdf = (
            gdf.buffer(5.0, join_style=1)
            .buffer(-5.0, join_style=1)
            .affine_transform(final_transform)
        )

        return gdf
