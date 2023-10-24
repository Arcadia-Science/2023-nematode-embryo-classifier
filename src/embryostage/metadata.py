import pandas as pd

from embryostage import file_utils


def load_dataset_metadata(dataset_id):
    '''
    Load the metadata for a given dataset as a pandas series
    '''
    repo_dirpath = file_utils.find_repo_root(__file__)
    dataset_metadata_filepath = repo_dirpath / "ground_truth" / "embryo_metadata.csv"
    if not dataset_metadata_filepath.exists():
        raise FileNotFoundError(f"Metadata file not found at {dataset_metadata_filepath}")

    dataset_metadata = pd.read_csv(dataset_metadata_filepath)

    # the dataset_id is a date of the from `YYMMDD` which is read as an integer
    dataset_metadata['dataset_id'] = dataset_metadata['dataset_id'].astype(str)

    dataset_metadata = dataset_metadata.loc[dataset_metadata.dataset_id == dataset_id]
    if dataset_metadata.empty:
        raise ValueError(f"Dataset {dataset_id} not found in the metadata file.")

    return dataset_metadata.squeeze()
