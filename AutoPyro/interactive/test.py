import os
from io import BytesIO

import holoviews as hv
import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.models import HoverTool
# Create a composite image with original and processed versions
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from skimage import color, exposure, filters, io, measure, morphology
from skimage.util import img_as_float, img_as_ubyte

pn.extension()
hv.extension("bokeh")


class ImageDigitizer(param.Parameterized):
    # Parameters
    image_file = param.FileSelector(path="./*.png;./*.jpg;./*.jpeg;./*.tif;./*.tiff")
    upload = param.FileSelector()
    threshold_method = param.ObjectSelector(
        default="otsu", objects=["otsu", "li", "yen", "isodata", "mean"]
    )
    threshold_adjust = param.Number(default=0.0, bounds=(-0.5, 0.5), step=0.01)
    invert_threshold = param.Boolean(default=False)
    min_object_size = param.Integer(default=100, bounds=(0, 10000))
    max_object_size = param.Integer(default=100000, bounds=(0, 1000000))
    apply_watershed = param.Boolean(default=False)
    watershed_markers = param.Integer(default=50, bounds=(10, 500))
    contrast_min = param.Number(default=0.0, bounds=(0.0, 1.0), step=0.01)
    contrast_max = param.Number(default=1.0, bounds=(0.0, 1.0), step=0.01)
    gamma = param.Number(default=1.0, bounds=(0.1, 3.0), step=0.1)

    # Processed data
    original_image = param.Array()
    processed_image = param.Array()
    binary_image = param.Array()
    labeled_image = param.Array()
    regions = param.List()

    def __init__(self, **params):
        super().__init__(**params)
        self._update_image()

    @param.depends("upload", watch=True)
    def _handle_upload(self):
        if self.upload:
            # Save uploaded file to disk
            if not os.path.exists("uploads"):
                os.makedirs("uploads")
            filepath = os.path.join("uploads", self.upload)
            with open(filepath, "wb") as f:
                f.write(self.upload)
            self.image_file = filepath

    @param.depends("image_file", watch=True)
    def _update_image(self):
        if self.image_file and os.path.exists(self.image_file):
            self.original_image = io.imread(self.image_file)
            self._process_image()

    @param.depends(
        "threshold_method",
        "threshold_adjust",
        "invert_threshold",
        "min_object_size",
        "max_object_size",
        "apply_watershed",
        "watershed_markers",
        "contrast_min",
        "contrast_max",
        "gamma",
        watch=True,
    )
    def _process_image(self):
        if self.original_image is None:
            return

        # Convert to grayscale if needed
        if len(self.original_image.shape) == 3:
            img = color.rgb2gray(self.original_image)
        else:
            img = self.original_image

        # Adjust contrast
        img = exposure.rescale_intensity(
            img, in_range=(self.contrast_min, self.contrast_max)
        )
        img = exposure.adjust_gamma(img, gamma=self.gamma)
        self.processed_image = img

        # Apply threshold
        threshold_func = {
            "otsu": filters.threshold_otsu,
            "li": filters.threshold_li,
            "yen": filters.threshold_yen,
            "isodata": filters.threshold_isodata,
            "mean": filters.threshold_mean,
        }[self.threshold_method]

        try:
            thresh = threshold_func(img)
        except:
            thresh = 0.5

        thresh = thresh + self.threshold_adjust
        binary = img > thresh

        if self.invert_threshold:
            binary = ~binary

        # Clean up binary image
        binary = morphology.remove_small_objects(binary, self.min_object_size)
        binary = morphology.remove_small_holes(binary, self.min_object_size)
        self.binary_image = binary

        # Apply watershed if needed
        if self.apply_watershed:
            distance = filters.sobel(binary)
            markers = morphology.label(binary)
            labeled = morphology.watershed(-distance, markers, mask=binary)
        else:
            labeled = measure.label(binary)

        # Filter objects by size
        regions = measure.regionprops(labeled)
        sizes = [r.area for r in regions]
        mask = (np.array(sizes) >= self.min_object_size) & (
            np.array(sizes) <= self.max_object_size
        )
        filtered_labels = np.zeros_like(labeled)

        for i, region in enumerate(regions):
            if mask[i]:
                filtered_labels[labeled == region.label] = region.label

        self.labeled_image = filtered_labels
        self.regions = [r for i, r in enumerate(regions) if mask[i]]

    def view_original(self):
        if self.original_image is None:
            return hv.RGB([])

        if len(self.original_image.shape) == 3:
            return hv.RGB(self.original_image)

        return hv.Image(self.original_image).opts(
            cmap="gray", title="Original Image"
        )

    def view_processed(self):
        if self.processed_image is None:
            return hv.RGB([])
        return hv.Image(self.processed_image).opts(cmap="gray", title="Processed Image")

    def view_binary(self):
        if self.binary_image is None:
            return hv.RGB([])
        return hv.Image(self.binary_image).opts(cmap="binary", title="Binary Image")

    def view_labeled(self):
        if self.labeled_image is None:
            return hv.RGB([])

        # Create a colorful overlay of labeled regions
        colored_labels = color.label2rgb(self.labeled_image, bg_label=0)
        overlay = hv.RGB(colored_labels).opts(title="Labeled Objects", alpha=0.5)

        # Add hover tool with region properties
        xs, ys = np.meshgrid(
            np.arange(self.labeled_image.shape[1]),
            np.arange(self.labeled_image.shape[0]),
        )
        points = hv.Points((xs.ravel(), ys.ravel(), self.labeled_image.ravel()))

        hover = HoverTool(tooltips=[("Label", "@{z}"), ("(x,y)", "($x, $y)")])

        return (overlay * points).opts(tools=[hover])

    def view_region_table(self):
        if not self.regions:
            return pn.pane.Markdown("No regions detected")

        data = []
        for region in self.regions:
            data.append(
                {
                    "Label": region.label,
                    "Area": region.area,
                    "Perimeter": region.perimeter,
                    "Centroid X": region.centroid[1],
                    "Centroid Y": region.centroid[0],
                    "Bounding Box": str(region.bbox),
                }
            )

        df = pd.DataFrame(data)
        return df.hvplot.table()

    def view_region_properties(self):
        if not self.regions:
            return pn.pane.Markdown("No regions detected")

        plots = []
        properties = ["area", "perimeter", "eccentricity", "solidity"]

        for prop in properties:
            values = [getattr(r, prop) for r in self.regions]
            if all(np.isfinite(values)):
                plots.append(
                    hv.Histogram(np.histogram(values)).opts(title=prop.capitalize())
                )

        return hv.Layout(plots).cols(2)

    def save_results(self):
        if self.labeled_image is None:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Original image
        if len(self.original_image.shape) == 3:
            axes[0, 0].imshow(self.original_image)
        else:
            axes[0, 0].imshow(self.original_image, cmap="gray")
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        # Processed image
        axes[0, 1].imshow(self.processed_image, cmap="gray")
        axes[0, 1].set_title("Processed Image")
        axes[0, 1].axis("off")

        # Binary image
        axes[1, 0].imshow(self.binary_image, cmap="binary")
        axes[1, 0].set_title("Binary Image")
        axes[1, 0].axis("off")

        # Labeled image
        axes[1, 1].imshow(color.label2rgb(self.labeled_image, bg_label=0))
        axes[1, 1].set_title("Labeled Objects")
        axes[1, 1].axis("off")

        plt.tight_layout()

        # Save to buffer
        buf = BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        plt.close(fig)
        buf.seek(0)

        return buf


# Create the app
digitizer = ImageDigitizer()

# Define the layout
file_selector = pn.Param(
    digitizer.param.image_file,
    widgets={"image_file": pn.widgets.FileSelector(size=10)},
    name="Select Image",
)

upload_widget = pn.widgets.FileInput(
    accept=".png,.jpg,.jpeg,.tif,.tiff", name="Upload Image"
)
upload_widget.link(digitizer, value="upload")

threshold_panel = pn.Param(
    digitizer,
    parameters=["threshold_method", "threshold_adjust", "invert_threshold"],
    name="Threshold Settings",
)

object_panel = pn.Param(
    digitizer,
    parameters=[
        "min_object_size",
        "max_object_size",
        "apply_watershed",
        "watershed_markers",
    ],
    name="Object Detection",
)

contrast_panel = pn.Param(
    digitizer,
    parameters=["contrast_min", "contrast_max", "gamma"],
    name="Contrast Adjustment",
)

image_views = pn.Tabs(
    ("Original", pn.panel(digitizer.view_original, width=600, height=600)),
    ("Processed", pn.panel(digitizer.view_processed, width=600, height=600)),
    ("Binary", pn.panel(digitizer.view_binary, width=600, height=600)),
    ("Labeled", pn.panel(digitizer.view_labeled, width=600, height=600)),
)

analysis_views = pn.Tabs(
    ("Region Table", pn.panel(digitizer.view_region_table, width=800)),
    ("Region Properties", pn.panel(digitizer.view_region_properties, width=800)),
)

download_button = pn.widgets.FileDownload(
    callback=digitizer.save_results,
    filename="digitization_results.png",
    button_type="primary",
    label="Download Results",
)

# Assemble the dashboard
app = pn.Column(
    pn.Row(
        pn.Column(
            pn.panel(upload_widget, width=300),
            file_selector,
            threshold_panel,
            object_panel,
            contrast_panel,
            download_button,
        ),
        image_views,
    ),
    analysis_views,
)

# Serve the app
app.servable(title="Image Digitization Tool")
