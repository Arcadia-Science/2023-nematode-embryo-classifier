from pathlib import Path
import click
import zarr

from tqdm import tqdm

from embryostage.cli import options as cli_options
from embryostage.preprocess.utils import compute_morphodynamics


@cli_options.data_dirpath_option
@cli_options.dataset_id_option
@click.command()
def encode_dynamics(data_dirpath, dataset_id):
    '''
    Call compute_morphodynamics() on all cropped embryos in a dataset and save the results
    '''

    dataset_dirpath = data_dirpath / "cropped_embryos" / dataset_id

    # the list of all FOV IDs in the dataset
    fov_ids = [dirpath.name for dirpath in dataset_dirpath.glob('fov*')]

    for fov_id in fov_ids:
        cropped_embryo_filepaths = list((dataset_dirpath / fov_id).glob("embryo-*"))
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
            zarr.save_group(Path(cropped_embryo_filepath, "dynamic_features"), **feature_dict)


if __name__ == '__main__':
    encode_dynamics()
