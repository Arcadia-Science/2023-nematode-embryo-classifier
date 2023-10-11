# %% imports
from pathlib import Path
from embryostage.preprocess.EmbryoFinder import EmbryoFinder

# %% Interactive execution.
if __name__ == "__main__":
    # Set all parameters here.
    topdir = "D:\\Matus\\C_elegans\\predict_development\\"
    input_path = Path(topdir, "ChannelBF_20x_Seq0021.zarr").expanduser()

    # Date stamp and FOVs to process.
    # Check the annotations for each FOV, if it is reanalyzed.
    FOVs = [str(i) for i in range(99)]

    # Parameters of the imaging experiment.
    xy_sampling = 0.43  # Sampling in um/pixel in the sample plane.
    t_sampling = 300  # Sampling in seconds/frame.
    l_embryo = 65  # Length of the embryo in um.
    d_embryo = 32.5  # Diameter of the embryo in um.

    # Biological parameters.
    output_path = Path(topdir, "celegans_movies").expanduser()
    date_stamp = "230719"
    strain = "DQM327"
    perturbation = "control"

    # viewer = napari.Viewer()
    # napari.run()

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
