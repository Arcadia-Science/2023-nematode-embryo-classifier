import pathlib
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent


def load_dataset_metadata(dataset_id):
    '''
    Load the metadata for a given dataset as a pandas series
    '''
    dataset_metadata_filepath = REPO_ROOT / "ground_truth" / "embryo_metadata.csv"
    if not dataset_metadata_filepath.exists():
        raise FileNotFoundError(f"Metadata file not found at {dataset_metadata_filepath}")

    dataset_metadata = pd.read_csv(dataset_metadata_filepath)
    dataset_metadata = dataset_metadata.loc[dataset_metadata.dataset_id == dataset_id]
    if dataset_metadata.empty:
        raise ValueError(f"Dataset {dataset_id} not found in the metadata file.")

    return dataset_metadata.squeeze()
