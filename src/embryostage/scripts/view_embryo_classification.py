from pathlib import Path
import click
import napari
import numpy as np
import torch
import zarr

from embryostage.cli import options as cli_options
from embryostage.models.classification import SulstonNet

channels_type_option = click.option(
    '--channels-type',
    type=str,
    help="The type of channel(s) used to train the model ('dynamic' or 'raw-only')",
)

checkpoint_filepath_option = click.option(
    '--checkpoint-filepath', type=str, help='The path to the model checkpoint (.ckpt) file'
)


@cli_options.data_dirpath_option
@cli_options.dataset_id_option
@click.option('--fov-id', type=str, help='The ID of the FOV')
@click.option('--embryo-id', type=str, help='The ID of the Embryo')
@checkpoint_filepath_option
@channels_type_option
@click.command()
def view_embryo_classification(
    data_dirpath, dataset_id, fov_id, embryo_id, channels_type, checkpoint_filepath
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

    embryo_filepath = (
        data_dirpath
        / 'encoded_dynamics'
        / dataset_id
        / f'fov{fov_id}'
        / f'embryo-{embryo_id}.zarr'
    )

    device = torch.device(device_name)

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
        viewer.text_overlay.text = f"prediction:{index_to_label[predictions[frame_ind]]}"

    # Connect the update function to the slider
    viewer.dims.events.current_step.connect(text_overlay)

    # Overlay settings
    viewer.text_overlay.color = "white"
    viewer.text_overlay.font_size = 18
    viewer.text_overlay.visible = True

    # Start the napari event loop
    napari.run()


if __name__ == '__main__':
    view_embryo_classification()
