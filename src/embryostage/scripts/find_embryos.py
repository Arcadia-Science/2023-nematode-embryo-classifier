from pathlib import Path
import click

from embryostage.metadata import load_dataset_metadata
from embryostage.preprocess.embryo_finder import EmbryoFinder


@click.option("--data-dirpath", type=Path, help="Path to the data directory")
@click.option("--dataset-id", type=str, help="The ID of the dataset to process")
@click.command()
def find_embryos(data_dirpath, dataset_id):
    '''
    This is a wrapper for calling EmbryoFinder.find_embryos()
    '''

    input_path = data_dirpath / 'raw_data' / dataset_id
    output_path = data_dirpath / 'cropped_embryos' / dataset_id

    if not input_path.exists():
        raise FileNotFoundError(
            f"No raw_data directory for dataset '{dataset_id}' found in {data_dirpath}"
        )

    # Load the metadata for the dataset
    dataset_metadata = load_dataset_metadata(dataset_id=dataset_id)

    # the list of all FOV IDs in the dataset
    fov_ids = [dirpath.name for dirpath in input_path.glob('fov*')]

    embryo_finder = EmbryoFinder(
        input_path=input_path,
        output_path=output_path,
        fov_ids=fov_ids,
        xy_sampling_um=float(dataset_metadata.xy_sampling_um),
        embryo_length_um=float(dataset_metadata.embryo_length_um),
        embryo_diameter_um=float(dataset_metadata.embryo_diameter_um),
    )

    embryo_finder.find_embryos()


if __name__ == '__main__':
    find_embryos()
