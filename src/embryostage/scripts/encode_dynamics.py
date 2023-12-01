from pathlib import Path
import click
import zarr

from tqdm import tqdm

from embryostage import cli_options
from embryostage.models import constants
from embryostage.preprocess.utils import compute_morphodynamics


@cli_options.data_dirpath_option
@cli_options.dataset_id_option
@click.command()
def main(data_dirpath, dataset_id):
    """
    Call compute_morphodynamics() on all cropped embryos in a dataset and save the results
    """

    input_path = data_dirpath / "cropped_embryos" / dataset_id
    output_path = data_dirpath / "encoded_dynamics" / dataset_id

    if not input_path.exists():
        raise FileNotFoundError(
            f"No cropped_embryos directory for dataset '{dataset_id}' found in {data_dirpath}"
        )

    # the list of all FOV IDs in the dataset
    fov_ids = [dirpath.name for dirpath in input_path.glob("fov*")]

    for fov_id in fov_ids:
        cropped_embryo_filepaths = list((input_path / fov_id).glob("embryo-*"))
        if not cropped_embryo_filepaths:
            print(f"No embryos found for FOV '{fov_id}' in dataset '{dataset_id}'")
            continue

        print(f"Computing morphodynamics for embryos from FOV '{fov_id}'")

        for cropped_embryo_filepath in tqdm(cropped_embryo_filepaths):
            cropped_embryo = zarr.open(str(cropped_embryo_filepath), mode="r")
            feature_images, features = compute_morphodynamics(cropped_embryo)

            feature_dict = {
                features[n]: feature_images[:, n, ...] for n in range(len(features))
            }

            encoded_dynamics_filepath = output_path / fov_id / cropped_embryo_filepath.name
            zarr.save_group(
                Path(encoded_dynamics_filepath, constants.FEATURES_GROUP_NAME), **feature_dict
            )


if __name__ == "__main__":
    main()
