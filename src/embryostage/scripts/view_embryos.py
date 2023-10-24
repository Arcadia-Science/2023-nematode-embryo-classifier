import pathlib
import click
import napari
import zarr

from embryostage.preprocess.utils import get_movie_paths


@click.option("--data-dirpath", type=pathlib.Path, help="Path to the data directory")
@click.option("--dataset-id", type=str, help="The ID of the dataset to process")
@click.command()
def view_embryos(data_dirpath, dataset_id):
    '''
    Load the timelapses for all embryos from one or more FOVs from a given dataset
    into napari for viewing and annotating the embryos.
    '''

    viewer = napari.Viewer()

    fov_ids = [0, 1, 2, 3]
    movie_paths = get_movie_paths(data_dirpath, dataset_id, fov_ids=list(map(str, fov_ids)))

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
