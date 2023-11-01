# This test parses the annotations of developmental stages and writes them
# with the corresponding zarr stores.
# Doing so checks that annotations and data match, and are correctly parsed.
# Run this test with pytest -s -v --disable_warings test_annotations.py
# Spot-check the annotations.csv file written to each zarr store.

from pathlib import Path
import pytest

from embryostage.models.data import link_annotations_to_zarrs

# TODO (KC): move this to a fixture
REPO_DIRPATH = Path(__file__).parent.parent


@pytest.mark.skip(reason="depends on a test dataset that does not exist yet")
def test_annotations():
    annotations_path = REPO_DIRPATH / "ground_truth" / "embryo_developmental_stage.csv"
    metadata_path = REPO_DIRPATH / "ground_truth" / "embryo_metadata.csv"

    # TODO (KC): replace this with a test dataset
    data_dirpath = Path(
        "~/docs/data/predict_development/celegans_embryos_dataset"
    ).expanduser()

    link_annotations_to_zarrs(
        data_dirpath=data_dirpath,
        annotations_path=annotations_path,
        metadata_path=metadata_path,
    )
