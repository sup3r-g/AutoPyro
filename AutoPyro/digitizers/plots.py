import os

import numpy as np
import rasterio as rio
from affine import Affine
from rasterio.plot import reshape_as_image
from scipy import ndimage as ndi
from skimage import color, exposure, filters, io, measure, morphology, transform
from sklearn.cluster import KMeans

from AutoPyro.digitizers.vector_files import SVGParse

class ImagePlotDigitizer:
    # __slots__ = "image", ""

    def __init__(
        self,
        image: np.ndarray,
        binary_image: np.ndarray,
        lines: list[np.ndarray],
        curves: list[np.ndarray],
        transform_points: list[np.ndarray],
    ) -> None:
        self.image = image
        self.binary_image = binary_image
        self.lines = lines
        self.curves = curves
        self.transform_points = transform_points

    @classmethod
    def from_file(cls, image_file: str | os.PathLike):
        image = io.imread(image_file)
        if image.shape[2] > 3:
            image = color.rgba2rgb(image)

        return cls(image)

    def from_svg(cls, svg_file: str | os.PathLike):
        return SVGParse(svg_file).svg2json()

    def _find_lines(self, angles):
        h, theta, d = transform.hough_line(self.binary_image, theta=angles)

        for i, (_, angle, dist) in enumerate(
            zip(*transform.hough_line_peaks(h, theta, d))
        ):
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            ang = np.tan(angle + np.pi / 2)
            if np.isclose(ang, 0, atol=0.01):
                ang = 0
            elif np.isinf(ang):
                pass

        return (x1, y1), (x2, y2)

    def create_binary(self, block_size: int = 10, offset: int = 10):
        gray_image = color.rgb2gray(self.image)
        local_thresh = filters.threshold_local(gray_image, block_size, offset=offset)
        self.binary_image = gray_image < local_thresh

    def find_axis_lines(self):
        x_angles = np.linspace(3 * np.pi / 8, 5 * np.pi / 8, 90, endpoint=True)
        y_angles = np.linspace(-np.pi / 8, np.pi / 8, 90, endpoint=False)
        other_angles = np.linspace(np.pi / 8, 3 * np.pi / 8, 180, endpoint=False)

        x_axis = self._find_lines(x_angles)
        y_axis = self._find_lines(y_angles)

    def find_curves(self):
        # contours = measure.find_contours(self.binary_image, 0.8)
        skeleton = morphology.skeletonize(self.binary_image)

        for contour in skeleton:
            pass
            segmentation.flood_fill(binary_inv, (contour[0][1]-1, contour[0][0]-1), 255)
            contour[:, 1], contour[:, 0]  x, y
