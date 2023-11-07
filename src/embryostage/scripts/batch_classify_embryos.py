import json
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


def parse_ids_from_embryo_filepath(embryo_filepath):
    '''
    parse the dataset ID, FOV ID, and embryo ID from an embryo filepath of the form
    /some/path/<dataset_id>/fov<fov_id>/embryo-<embryo_id>.zarr
    '''
    embryo_filepath = Path(embryo_filepath)
    dataset_id = embryo_filepath.parent.parent.name
    fov_id = embryo_filepath.parent.name.replace('fov', '')
    embryo_id = embryo_filepath.stem.replace('embryo-', '')
    return {'dataset_id': dataset_id, 'fov_id': fov_id, 'embryo_id': embryo_id}


@cli_options.data_dirpath_option
@cli_options.dataset_id_option
@click.option(
    '--channels-type',
    type=str,
    help="The type of channel(s) used to train the model ('dynamic' or 'raw')",
)
@click.option(
    '--checkpoint-filepath',
    type=Path,
    help='The path to the model checkpoint (.ckpt) file',
)
@click.option(
    '--device-name',
    type=str,
    default='cpu',
    help='The device on which to run inference ("cpu", "cuda", or "mps")',
)
@click.command()
def main(data_dirpath, dataset_id, channels_type, checkpoint_filepath, device_name):
    '''
    predict the classification for all embryos in a dataset
    '''

    if channels_type == "dynamic":
        channel_names = ["moving_mean", "moving_std"]
    elif channels_type == "raw":
        channel_names = ["raw"]
    else:
        raise ValueError(f"Invalid channel type '{channels_type}'. Must be 'moving' or 'raw'.")

    # the name of the zarr group containing the encoded dynamics (or 'features')
    # note: this group includes a copy of the raw images,
    # so the group name does not depend on `channels_type`
    zarr_group_name = constants.FEATURES_GROUP_NAME

    # the x-y size of the cropped embryos
    image_shape = (constants.EMBRYO_CROP_SIZE, constants.EMBRYO_CROP_SIZE)

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

        logits = logits.to("cpu").numpy()
        predicted_label_inds = predicted_label_inds.to("cpu").numpy()

        predicted_labels = [
            constants.EMBRYO_STAGE_INDEX_TO_LABEL[ind] for ind in predicted_label_inds
        ]

        all_predicted_labels.append(
            {
                "logits": logits.tolist(),
                "labels": predicted_labels,
                "embryo_filepath": str(embryo_filepath),
                **parse_ids_from_embryo_filepath(embryo_filepath),
            }
        )

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d")

    # write the predictions to a CSV file in the checkpoint's parent directory
    # (this is a hackish way to associate the predictions with the model that generated them)
    output_filepath = (
        checkpoint_filepath.parent.parent
        / f'{timestamp}-preds-for-{dataset_id}--from-{checkpoint_filepath.stem}.json'
    )

    os.makedirs(output_filepath.parent, exist_ok=True)
    with open(output_filepath, 'w') as file:
        json.dump(all_predicted_labels, file)

    print(f"Predictions saved to {output_filepath}")


if __name__ == '__main__':
    main()
