import click
import napari
import zarr

from embryostage.cli import options as cli_options
from embryostage.preprocess.utils import get_movie_paths


@cli_options.data_dirpath_option
@cli_options.dataset_id_option
@click.option("--fov-ids", type=str, help="The IDs of the FOVs as a comma-separated list")
@click.command()
def view_embryos(data_dirpath, dataset_id, fov_ids):
    '''
    Load the timelapses for all embryos from one or more FOVs from a given dataset
    into napari for viewing and annotating the embryos.
    '''
    viewer = napari.Viewer()

    fov_ids = [fov_id.strip() for fov_id in fov_ids.split(",")]
    movie_paths = get_movie_paths(data_dirpath, dataset_id, fov_ids=fov_ids)

    for movie_path in movie_paths:
        movie = zarr.open(str(movie_path), mode="r")
        fov_id = movie_path.parent.name
        embryo_id = movie_path.name
        viewer.add_image(
            movie, name=f"{fov_id}_{embryo_id}", colormap="gray", blending="opaque"
        )

    napari.run()


if __name__ == '__main__':
    view_embryos()
