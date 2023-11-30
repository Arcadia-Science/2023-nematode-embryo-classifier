import click
import napari
import zarr

from embryostage import cli_options
from embryostage.preprocess.utils import get_cropped_embryo_filepaths


@cli_options.data_dirpath_option
@cli_options.dataset_id_option
@click.option("--fov-ids", type=str, help="The IDs of the FOVs as a comma-separated list")
@click.command()
def main(data_dirpath, dataset_id, fov_ids):
    """
    Load the timelapses for all embryos from one or more FOVs from a given dataset
    into napari for viewing and annotating the embryos.
    """
    viewer = napari.Viewer()

    fov_ids = [fov_id.strip() for fov_id in fov_ids.split(",")]
    cropped_embryo_filepaths = get_cropped_embryo_filepaths(
        data_dirpath, dataset_id, fov_ids=fov_ids
    )

    for cropped_embryo_filepath in cropped_embryo_filepaths:
        cropped_embryo = zarr.open(str(cropped_embryo_filepath), mode="r")
        fov_id = cropped_embryo_filepath.parent.name
        embryo_id = cropped_embryo_filepath.name
        viewer.add_image(
            cropped_embryo, name=f"{fov_id}_{embryo_id}", colormap="gray", blending="opaque"
        )

    napari.run()


if __name__ == "__main__":
    main()
