import pathlib
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent


def load_dataset_metadata(dataset_id):
    '''
    Load the metadata for a given dataset as a pandas series
    '''
    dataset_metadata_filepath = REPO_ROOT / "ground_truth" / "embryo_metadata.csv"
    dataset_metadata = pd.read_csv(dataset_metadata_filepath)

    # HACK: for now, we use the date as the dataset_id
    dataset_metadata["dataset_id"] = dataset_metadata["date"].apply(str)

    if dataset_id not in dataset_metadata.dataset_id.values:
        raise ValueError(f"Dataset {dataset_id} not found in the metadata file.")

    dataset_metadata = dataset_metadata.loc[dataset_metadata.dataset_id == dataset_id].iloc[0]
    return dataset_metadata
