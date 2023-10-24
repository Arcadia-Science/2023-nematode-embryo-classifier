# %% Imports and classes.
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import zarr

from iohub.ngff import open_ome_zarr
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion, disk
from skimage.segmentation import clear_border


class EmbryoFinder:
    """
    Class for finding and cropping C. elegans embryos in a time series.

    Attributes
    ----------
    input_path : str
        Path to the input zarr store.
    fov_ids : list of str
        List of field of view (FOV) IDs.
    output_path : str
        Path to the output zarr store.
    xy_sampling_um : int
        Sampling rate in the x and y dimensions in microns
    t_sampling_sec : int
        Sampling rate in the time dimension in seconds.
    embryo_length_um : int
        Length of the embryo in microns.
    embryo_diameter_um : int
        Diameter of the embryo in microns.

    Methods
    -------
    find_embryos()
        Find and crop C. elegans embryos in the time series.
        Results are stored in `{output_path}/fov{fov_id}/embryo-{embryo_id}.zarr`

        The embryo*.zarr stores can be dragged into napari.
        Multiple FOVs can be visualized using the view_embryos CLI.

    segment_time_projection()
        Segment the time projection of the input data.
    """

    def __init__(
        self,
        input_path,
        fov_ids,
        xy_sampling_um,
        t_sampling_sec,
        embryo_length_um,
        embryo_diameter_um,
        output_path,
    ):
        """
        Initialize EmbryoFinder object.

        Parameters
        ----------
        input_path : str (created with bioformats2raw).
            Path to the input zarr store.
        fov_ids : list of str
            List of field of view (FOV) IDs (typically numbers).
        xy_sampling_um : int
            Pixel size in the x and y dimensions in microns/pixel.
        t_sampling_sec : int
            Sampling interval in the time dimension in seconds/frame.
        embryo_length_um : int
            Length of the embryo in microns.
        embryo_diameter_um : int
            Diameter of the embryo in microns.
        output_path : str
            Path to the output zarr store.

        """
        self.input_path = input_path
        self.fov_ids = fov_ids
        self.output_path = output_path
        self.xy_sampling_um = xy_sampling_um
        self.t_sampling_sec = t_sampling_sec
        self._embryo_length_um = embryo_length_um
        self._embryo_diameter_um = embryo_diameter_um

    @property
    def embryo_length_pix(self):
        return self._embryo_length_um // self.xy_sampling_um

    @property
    def embryo_diameter_pix(self):
        return self._embryo_diameter_um // self.xy_sampling_um

    @staticmethod
    def bounding_box_to_id(bounding_box):
        '''
        Crude way to generate an ID from bounding box coordinates that is unique within an FOV
        '''
        x_center = (bounding_box['xmin'] + bounding_box['xmax']) // 2
        y_center = (bounding_box['ymin'] + bounding_box['ymax']) // 2
        return f'{x_center:04d}-{y_center:04d}'

    def find_embryos(self):
        """
        Find and crop C. elegans embryos in the time series.
        """

        # Load the time series data and compute the time projection.
        for fov_id in self.fov_ids:
            # the path to the raw zarr store for the current FOV
            zarr_path = self.input_path / fov_id

            # the path to the output directory for the current FOV
            # (containing the cropped embryo zarrs and detected_embryos.png)
            fov_output_path = self.output_path / fov_id

            if not zarr_path.exists():
                print(f"Path {zarr_path} does not exist. Skipping FOV {fov_id}.")
                continue

            # Load the time series data and compute the time projection.
            print(f"Processing FOV {fov_id}.")
            with open_ome_zarr(
                Path(self.input_path, f"{fov_id}"),
                layout="fov",
                mode="r",
                channel_names="BF20x",
            ) as input_store:
                time_series = np.asarray(input_store['raw'])
                time_series_std = np.std(time_series, axis=0).squeeze()

                # Smoothing avoids detection of very small objects and
                # increases the size of the bounding box.
                smooth_projection = gaussian(
                    time_series_std, sigma=self.embryo_diameter_pix // 10
                )

                (embryo_bounding_boxes, mask, regions) = self.segment_time_projection(
                    smooth_projection, method="otsu"
                )

                self._plot_results(
                    method="otsu",
                    fov_id=fov_id,
                    embryo_bounding_boxes=embryo_bounding_boxes,
                    time_series_std=time_series_std,
                    smooth_projection=smooth_projection,
                    mask=mask,
                    export_path=fov_output_path,
                )

            # Save the cropped embryos to the output zarr store
            for _, embryo_bounding_box in enumerate(embryo_bounding_boxes):
                # generate an FOV-unique ID for the embryo
                embryo_id = self.bounding_box_to_id(embryo_bounding_box)
                embryo_path = fov_output_path / f'embryo-{embryo_id}.zarr'

                output_shape = (
                    time_series.shape[0],
                    time_series.shape[1],
                    time_series.shape[2],
                    int(self.embryo_length_pix - 1),
                    int(self.embryo_length_pix - 1),
                )

                output_store = zarr.creation.open_array(
                    embryo_path,
                    mode="w",
                    shape=output_shape,
                    chunks=output_shape,
                    dtype=time_series.dtype,
                )

                cropped_series = time_series[
                    :,
                    :,
                    :,
                    embryo_bounding_box["ymin"] : embryo_bounding_box["ymax"],
                    embryo_bounding_box["xmin"] : embryo_bounding_box["xmax"],
                ]

                output_store[:] = cropped_series

    def _plot_results(
        self,
        method,
        fov_id,
        embryo_bounding_boxes,
        time_series_std,
        smooth_projection,
        mask,
        export_path,
    ):
        """visualize the analysis results generated in find_embryos"""
        # TODO: turn this into a napari layer or widget.

        if method == "otsu":
            plt.figure(1).clear()
            plt.figure(1).set_size_inches(20, 5)
            plt.draw()

            plt.subplot(1, 3, 1)
            plt.imshow(time_series_std, cmap="gray")
            plt.title("std along time")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(smooth_projection, cmap="gray")
            plt.title("smoothed in xy")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(mask, cmap="gray")
            plt.title("segmented embryo candidates")

            plt.axis("off")

            for ind in range(len(embryo_bounding_boxes)):
                plt.text(
                    embryo_bounding_boxes[ind]["xmin"],
                    embryo_bounding_boxes[ind]["ymin"],
                    str(ind),
                    color="green",
                    fontsize=12,
                )

            plt.title(f"{len(embryo_bounding_boxes)} embryos detected")
            plt.draw()
            plt.pause(0.001)

            export_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(export_path / "detected_embryos.png")

        else:
            raise NotImplementedError

    def segment_time_projection(self, time_projection, method="otsu"):
        """
        Segment the time projection of the input data.

        Parameters
        ----------
        time_projection : numpy.ndarray
            Time projection of the input data.
        method : str, optional
            Segmentation method to use. Currently only "otsu" is supported.

        Returns
        -------
        embryos_fov : list of dict
            List of bounding boxes for the detected embryos.
        mask : numpy.ndarray
            Binary mask of the segmented embryos.
        regions : list of skimage.measure._regionprops._RegionProperties
            List of region properties for the segmented embryos.
        """
        if method == "otsu":
            binary = time_projection > threshold_otsu(time_projection)
        elif method == "hough":
            pass

        # Remove small objects and clear the border.
        binary = clear_border(binary)
        binary = binary_erosion(binary, disk(5))
        binary = binary_dilation(binary, disk(5))

        # Compute the bounding box around each embryo
        labeled = label(binary)
        regions = regionprops(labeled)

        embryo_bounding_boxes = []
        for n in range(len(regions)):
            # TODO (KC): document the origin of these magic numbers
            looks_like_embryo = (
                (
                    0.5 * self.embryo_length_pix
                    <= regions[n].major_axis_length
                    <= self.embryo_length_pix
                )
                and (
                    0.5 * self.embryo_length_pix * self.embryo_diameter_pix
                    <= regions[n].area
                    <= 0.8 * self.embryo_length_pix * self.embryo_diameter_pix
                )
                and (0.5 <= regions[n].eccentricity <= 0.95)
            )

            # Ensure that uniform square crops are taken around the embryo.
            center_y = (regions[n].bbox[0] + regions[n].bbox[2]) // 2
            center_x = (regions[n].bbox[1] + regions[n].bbox[3]) // 2

            xmin = int(center_x - self.embryo_length_pix // 2)
            xmax = int(center_x + self.embryo_length_pix // 2)
            ymin = int(center_y - self.embryo_length_pix // 2)
            ymax = int(center_y + self.embryo_length_pix // 2)

            embryo_within_image = (
                xmin >= 0
                and ymin >= 0
                and ymax <= time_projection.shape[0]
                and xmax <= time_projection.shape[1]
            )

            if looks_like_embryo and embryo_within_image:
                embryo_bounding_boxes.append(
                    {"ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax}
                )

        return (embryo_bounding_boxes, binary, regions)
