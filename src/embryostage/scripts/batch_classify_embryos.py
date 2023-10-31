import os

from pathlib import Path
import click
import numpy as np
import pandas as pd
import torch
import zarr

from tqdm import tqdm

from embryostage.cli import options as cli_options
from embryostage.models.classification import SulstonNet
from embryostage.scripts.view_embryo_classification import (
    channels_type_option,
    checkpoint_filepath_option,
)


@cli_options.data_dirpath_option
@cli_options.dataset_id_option
@channels_type_option
@checkpoint_filepath_option
@click.command()
def batch_classify_embryos(data_dirpath, dataset_id, channels_type, checkpoint_filepath):
    '''
    predict the classification for all embryos in a dataset
    '''

    if channels_type == "dynamic":
        channel_names = ["moving_mean", "moving_std"]
    elif channels_type == "raw-only":
        channel_names = ["raw"]
    else:
        raise ValueError(f"Invalid channel type '{channels_type}'. Must be 'moving' or 'raw'.")

    index_to_label = {
        0: "proliferation",
        1: "bean",
        2: "comma",
        3: "fold",
        4: "hatch",
        5: "death",
        # 6: "unfertilized",
    }

    device_name = "mps"
    zarr_group_name = "dynamic_features"

    # the x-y size of the cropped embryos
    image_shape = (224, 224)

    trained_model = SulstonNet.load_from_checkpoint(
        checkpoint_filepath,
        in_channels=len(channel_names),
        n_classes=len(index_to_label),
        index_to_label=index_to_label,
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
            predicted_labels = torch.argmax(logits, axis=1)

        predicted_labels = predicted_labels.to("cpu").numpy()
        predicted_labels = [index_to_label[label] for label in predicted_labels]
        all_predicted_labels.append(
            pd.Series(
                predicted_labels, name=f"{embryo_filepath.parent.name}_{embryo_filepath.name}"
            )
        )

    all_predicted_labels = pd.DataFrame(all_predicted_labels)

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    csv_path = (
        data_dirpath
        / 'predicted_classifications'
        / f'{timestamp}--dataset-{dataset_id}--classifications.csv'
    )

    os.makedirs(csv_path.parent, exist_ok=True)
    all_predicted_labels.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")


if __name__ == '__main__':
    batch_classify_embryos()
