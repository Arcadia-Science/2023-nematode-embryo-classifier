import pathlib
import click

data_dirpath_option = click.option(
    "--data-dirpath", type=pathlib.Path, help="Path to the data directory"
)
dataset_id_option = click.option(
    "--dataset-id", type=str, help="The ID of the dataset to process"
)
