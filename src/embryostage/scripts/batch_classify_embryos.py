import os

from pathlib import Path
import click
import numpy as np
import pandas as pd
import torch
import zarr

from tqdm import tqdm

from embryostage.cli import options as cli_options
from embryostage.models import constants
from embryostage.models.classification import SulstonNet


@cli_options.data_dirpath_option
@cli_options.dataset_id_option
@click.option(
    '--channels-type',
    type=str,
    help="The type of channel(s) used to train the model ('dynamic' or 'raw-only')",
)
@click.option(
    '--checkpoint-filepath',
    type=Path,
    help='The path to the model checkpoint (.ckpt) file',
)
@click.command()
def main(data_dirpath, dataset_id, channels_type, checkpoint_filepath):
    '''
    predict the classification for all embryos in a dataset
    '''

    if channels_type == "dynamic":
        channel_names = ["moving_mean", "moving_std"]
    elif channels_type == "raw-only":
        channel_names = ["raw"]
    else:
        raise ValueError(f"Invalid channel type '{channels_type}'. Must be 'moving' or 'raw'.")

    device_name = "mps"
    zarr_group_name = "dynamic_features"

    # the x-y size of the cropped embryos
    image_shape = (224, 224)

    trained_model = SulstonNet.load_from_checkpoint(
        checkpoint_filepath,
        n_input_channels=len(channel_names),
        n_classes=len(constants.EMBRYO_STAGE_INDEX_TO_LABEL),
        index_to_label=constants.EMBRYO_STAGE_INDEX_TO_LABEL,
    )

    # Make sure the model is in eval mode for inference
    trained_model.eval()
    device = torch.device(device_name)

    # aggregate filepaths to the encoded dynamics for all embryos from all FOVs in the dataset
    embryo_filepaths = list(
        (data_dirpath / 'encoded_dynamics' / dataset_id).glob('fov*/embryo*.zarr')
    )

    if not embryo_filepaths:
        raise FileNotFoundError(
            f"No encoded dynamics data found for dataset '{dataset_id}' in {data_dirpath}"
        )

    all_predicted_labels = []
    for embryo_filepath in tqdm(embryo_filepaths):
        channel_images = []
        for channel_name in channel_names:
            channel_image = zarr.open(
                Path(embryo_filepath, zarr_group_name, channel_name), mode="r"
            )

            # First and last two frames are black. The array is in (T, C, H, W) shape.
            channel_image = np.array(channel_image[2:-2, ...])
            channel_images.append(channel_image)

        input_tensor = torch.from_numpy(np.concatenate(channel_images, axis=1))
        input_tensor = input_tensor.to(device)
        input_tensor = torch.nn.functional.interpolate(input_tensor, size=image_shape)

        # Run inference
        with torch.no_grad():
            logits = trained_model(input_tensor)
            predicted_label_inds = torch.argmax(logits, axis=1)

        predicted_label_inds = predicted_label_inds.to("cpu").numpy()
        predicted_labels = [
            constants.EMBRYO_STAGE_INDEX_TO_LABEL[ind] for ind in predicted_label_inds
        ]
        all_predicted_labels.append(
            pd.Series(
                predicted_labels, name=f"{embryo_filepath.parent.name}_{embryo_filepath.name}"
            )
        )

    all_predicted_labels = pd.DataFrame(all_predicted_labels)

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d")

    # write the predictions to a CSV file in the checkpoint's parent directory
    # (this is a hackish way to associate the predictions with the model that generated them)
    csv_path = (
        checkpoint_filepath.parent.parent
        / f'{timestamp}-predictions--from-{checkpoint_filepath.stem}--for-{dataset_id}.csv'
    )

    os.makedirs(csv_path.parent, exist_ok=True)
    all_predicted_labels.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")


if __name__ == '__main__':
    main()
