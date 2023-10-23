# %% imports
from pathlib import Path

from embryostage.preprocess.EmbryoFinder import EmbryoFinder


# %% Set all parameters here.

# Paths to the input and output folders.
topdir = "/mnt/embryostage-local/celegans_embryos_dataset"
input_path = Path(topdir, "230817_N2_heatshock_raw.zarr").expanduser()
output_path = Path(topdir, "230817_N2_heatshock_test").expanduser()


# FOVs to process.
# Check the annotations for each FOV, if it is reanalyzed.
FOVs = [f"fov{i}" for i in range(10)]

# Parameters of the imaging experiment.
xy_sampling = 0.22  # Sampling in um/pixel in the sample plane.
t_sampling = 300  # Sampling in seconds/frame.
l_embryo = 65  # Length of the embryo in um.
d_embryo = 32.5  # Diameter of the embryo in um.

# Biological parameters.
date_stamp = "230817"
strain = "N2"
perturbation = "heatshock"

# viewer = napari.Viewer()
# napari.run()

# %% Run the EmbryoFinder.
embryo_finder = EmbryoFinder(
    input_path,
    date_stamp,
    FOVs,
    xy_sampling,
    t_sampling,
    l_embryo,
    d_embryo,
    output_path,
    strain,
    perturbation,
)

embryo_finder.find_embryos()

# %%
