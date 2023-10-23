from pathlib import Path

from embryostage.preprocess.embryo_finder import EmbryoFinder

if __name__ == "__main__":
    topdir = "/mnt/embryostage-local/celegans_embryos_dataset"
    input_path = Path(topdir, "230817_N2_heatshock_raw.zarr").expanduser()
    output_path = Path(topdir, "230817_N2_heatshock_test").expanduser()

    # Biological parameters.
    date_stamp = "230817"
    strain = "N2"
    perturbation = "heatshock"

    # Date stamp and FOVs to process.
    # Check the annotations for each FOV, if it is reanalyzed.
    fov_ids = [str(ind) for ind in range(99)]

    # Parameters of the imaging experiment.
    # Sampling in um/pixel in the sample plane
    xy_sampling = 0.43

    # Sampling in seconds/frame
    t_sampling = 300

    # Length and diameter of the embryo in microns
    embryo_length_um = 65
    embryo_diameter_um = 32.5

    # Results are stored in
    # <output_path>/<strain>/<perturbation>/<date_stamp>_<fov>/<embryoN>.zarr.
    embryo_finder = EmbryoFinder(
        input_path=input_path,
        date_stamp=date_stamp,
        fov_ids=fov_ids,
        xy_sampling=xy_sampling,
        t_sampling=t_sampling,
        embryo_length_um=embryo_length_um,
        embryo_diamenter_um=embryo_diameter_um,
        output_path=output_path,
        strain=strain,
        perturbation=perturbation,
    )

    embryo_finder.find_embryos()
