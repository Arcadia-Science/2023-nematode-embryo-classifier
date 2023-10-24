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

    input_path = data_dirpath / dataset_id / 'raw_data'
    output_path = data_dirpath / dataset_id / 'cropped_embryos'

    if not input_path.exists():
        raise FileNotFoundError(
            f"No directory for dataset '{dataset_id}' found in {input_path}"
        )

    # Load the metadata for the dataset
    dataset_metadata = load_dataset_metadata(dataset_id=dataset_id)

    # hard-coded list of FOV IDs (assumes there are at most 99 FOVs)
    # TODO (KC): get the fov_ids from the metadata
    fov_ids = [dirpath.name for dirpath in input_path.glob('fov*')]

    embryo_finder = EmbryoFinder(
        input_path=input_path,
        output_path=output_path,
        fov_ids=fov_ids,
        xy_sampling_um=float(dataset_metadata.xy_sampling_um),
        t_sampling_sec=int(dataset_metadata.t_sampling_sec),
        embryo_length_um=float(dataset_metadata.embryo_length_um),
        embryo_diameter_um=float(dataset_metadata.embryo_diameter_um),
    )

    embryo_finder.find_embryos()


if __name__ == '__main__':
    find_embryos()
