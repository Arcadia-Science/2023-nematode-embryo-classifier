# %% Imports and parameters
import os
import napari
import zarr

from embryostage.preprocess.utils import get_movie_paths

# FIXME: althought napari can be launched from the terminal,
# it does not work from this script.
# The environment is setup correctly, but it looks like that the thread created by napari
# doesn't inherit the environment variables.

print(
    f'CONDA_PREFIX:{os.environ["CONDA_PREFIX"]}\n'
    f'QT_PLUGIN_PATH:{os.environ["QT_PLUGIN_PATH"]}\n'
    f'QT_QPA_PLATFORM:{os.environ["QT_QPA_PLATFORM"]}\n'
)


# %% Make a list of all the movies.

database_path = "/mnt/embryostage-local/celegans_embryos_dataset"
strain = "DQM327"
perturbation = "control"
date_stamp = "230719"
FOVs = [4, 5, 25]

movie_paths = get_movie_paths(database_path, strain, perturbation, date_stamp, FOVs)

viewer = napari.Viewer()

# %% Load movies as layers

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
