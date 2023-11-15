from pathlib import Path
import click
import napari
import numpy as np
import torch
import zarr

from embryostage import cli_options
from embryostage.models import constants
from embryostage.models.classification import SulstonNet


@cli_options.data_dirpath_option
@cli_options.dataset_id_option
@click.option('--fov-id', type=str)
@click.option(
    '--embryo-id',
    type=str,
    default='0',
    help='either a linear index or a true embryo ID of the form {x_cen}-{y_cen}',
)
@click.option(
    '--channels-type',
    type=str,
    help="The type of channel(s) used to train the model ('dynamic' or 'raw-only')",
)
@click.option(
    '--checkpoint-filepath', type=str, help='The path to the model checkpoint (.ckpt) file'
)
@click.command()
@click.option(
    '--device-name',
    type=str,
    default='cpu',
    help='The device on which to run inference ("cpu", "cuda", or "mps")',
)
def main(
    data_dirpath,
    dataset_id,
    fov_id,
    embryo_id,
    channels_type,
    checkpoint_filepath,
    device_name,
):
    '''
    interactively view the predicted classification
    for each frame of an embryo timelapse in napari
    '''

    if channels_type == "dynamic":
        channel_names = ["moving_mean", "moving_std"]
    elif channels_type == "raw-only":
        channel_names = ["raw"]
    else:
        raise ValueError(f"Invalid channel type '{channels_type}'. Must be 'moving' or 'raw'.")

    zarr_group_name = constants.FEATURES_GROUP_NAME

    # the x-y size of the cropped embryos
    image_shape = (constants.EMBRYO_CROP_SIZE, constants.EMBRYO_CROP_SIZE)

    trained_model = SulstonNet.load_from_checkpoint(
        checkpoint_filepath,
        n_input_channels=len(channel_names),
        n_classes=len(constants.EMBRYO_STAGE_LABELS),
        index_to_label=constants.EMBRYO_STAGE_INDEX_TO_LABEL,
    )

    # Make sure the model is in eval mode for inference
    trained_model.eval()
    device = torch.device(device_name)

    # if the embryo ID is a digit, assume it is a linear index
    if embryo_id.isdigit():
        embryo_filepaths = list(
            (data_dirpath / 'encoded_dynamics' / dataset_id / f'fov{fov_id}').glob(
                'embryo*.zarr'
            )
        )
        if not embryo_filepaths:
            raise FileNotFoundError(
                f"No encoded dynamics data found for dataset '{dataset_id}' and FOV '{fov_id}'"
            )

        if int(embryo_id) >= len(embryo_filepaths):
            raise ValueError(
                f"Invalid embryo index '{embryo_id}'. "
                f"There are only {len(embryo_filepaths)} embryos in FOV '{fov_id}'"
            )
        embryo_filepath = sorted(embryo_filepaths)[int(embryo_id)]

    # assume the embryo ID is a 'true' embryo_id of the form '{x_cen}-{y_cen}'
    else:
        embryo_filepath = (
            data_dirpath
            / 'encoded_dynamics'
            / dataset_id
            / f'fov{fov_id}'
            / f'embryo-{embryo_id}.zarr'
        )

    channel_images = []
    for channel_name in channel_names:
        print(f"Loading channel '{channel_name}' from '{embryo_filepath}'")
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
        predictions = torch.argmax(logits, axis=1)

    predictions = predictions.to("cpu").numpy()

    # Add the raw channel to input channels for display
    raw_channel = zarr.open(Path(embryo_filepath, zarr_group_name, "raw"), mode="r")
    raw_channel = np.array(raw_channel[2:-2, ...])
    displayed_images = np.concatenate((channel_images + [raw_channel]), axis=1)

    # Show the timelapse with predictions overlaid.
    viewer = napari.Viewer()
    viewer.add_image(
        displayed_images, name=f"{embryo_filepath.parent.name} / {embryo_filepath.name}"
    )

    def text_overlay():
        frame_ind = viewer.dims.current_step[0]
        viewer.text_overlay.text = (
            f"Prediction: {constants.EMBRYO_STAGE_INDEX_TO_LABEL[predictions[frame_ind]]}"
        )

    # Connect the update function to the slider
    viewer.dims.events.current_step.connect(text_overlay)

    # Overlay settings
    viewer.text_overlay.color = "white"
    viewer.text_overlay.font_size = 18
    viewer.text_overlay.visible = True

    # Start the napari event loop
    napari.run()


if __name__ == '__main__':
    main()
