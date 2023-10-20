# This test parses the annotations of developmental stages and writes them with the corresponding zarr stores.
# Doing so checks that annotations and data match, and are correctly parsed.
# Run this test with pytest -s -v --disable_warings test_annotations.py
# Spot-check the annotations.csv file written to each zarr store.

from pathlib import Path

from embryostage.models.data import link_annotations2zarrs


def test_annotations():
    annotations_path = Path(
        "../../../ground_truth/embryo_developmental_stage.csv"
    ).expanduser()

    database_path = Path(
        "~/docs/data/predict_development/celegans_embryos_dataset"
    ).expanduser()

    metadata_path = Path("../../../ground_truth/embryo_metadata.csv").expanduser()

    link_annotations2zarrs(
        annotations_path,
        database_path,
        metadata_path,
    )
