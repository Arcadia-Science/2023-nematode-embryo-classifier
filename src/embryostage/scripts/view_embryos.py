# %% Imports and parameters
from pathlib import Path
import napari
import zarr

from embryostage.preprocess.utils import get_movie_paths

# import os

# conda_prefix = os.environ["CONDA_PREFIX"]
# os.environ["QT_PLUGIN_PATH"] = f"{conda_prefix}/plugins"
# os.environ["QT_PLUGIN_PATH"] = "xcb"
# os.environ["DISPLAY"] = ":1.0"
# %% Make a list of all the movies.

# %% Load movies as layers

if __name__ == "__main__":
    database_path = "~/predict_development/celegans_embryos_dataset"
    strain = "DQM327"
    perturbation = "control"
    date_stamp = "230719"
    FOVs = [4, 5, 25]

    movie_paths = get_movie_paths(database_path, strain, perturbation, date_stamp, FOVs)

    viewer = napari.Viewer()

    for movie_path in movie_paths:
        # movie = open_ome_zarr(movie_path)[pyramid_level]
        movie = zarr.open(str(movie_path), mode="r")
        viewer.add_image(
            movie,
            name=f"{movie_path.parent.stem}_{movie_path.stem}",
            colormap="gray",
            blending="opaque",
        )

    napari.run()

# %%
