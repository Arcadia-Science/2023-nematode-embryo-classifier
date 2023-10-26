import pandas as pd

from embryostage import file_utils


def get_dataset_metadata_filepath():
    '''
    the manually-maintained CSV file of dataset metadata
    '''
    filepath = file_utils.find_repo_root(__file__) / "ground_truth" / "embryo_metadata.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset metadata file not found at {filepath}")
    return filepath


def get_annotations_filepath():
    '''
    the CSV of manual annotations of embryo developmental stages
    '''
    filepath = (
        file_utils.find_repo_root(__file__) / "ground_truth" / "embryo_developmental_stage.csv"
    )
    if not filepath.exists():
        raise FileNotFoundError(f"Manual annotations file not found at {filepath}")
    return filepath


def load_dataset_metadata(dataset_id):
    '''
    Load the metadata for a given dataset as a pandas series
    '''
    dataset_metadata_filepath = get_dataset_metadata_filepath()
    dataset_metadata = pd.read_csv(dataset_metadata_filepath)

    # the dataset_id is a date of the form `YYMMDD` which is read as an integer
    dataset_metadata['dataset_id'] = dataset_metadata['dataset_id'].astype(str)

    dataset_metadata = dataset_metadata.loc[dataset_metadata.dataset_id == dataset_id]
    if dataset_metadata.empty:
        raise ValueError(f"Dataset {dataset_id} not found in the metadata file.")

    return dataset_metadata.squeeze()
