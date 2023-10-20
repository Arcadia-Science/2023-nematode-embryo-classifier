from pathlib import Path
import napari
import numpy as np
import torch
import zarr

from embryostage.models.classification import SulstonNet

# Load a trained model from a checkpoint
checkpoint_path = (
    "~/data/predict_development/celegans_embryos_dataset/models/lightning_logs/"
    "sulstonNet_heatshock_7classes_moving_mean_std/checkpoints/"
    "checkpoint-epoch=10-val_loss=0.10.ckpt"
)

channel_names = ["moving_mean", "moving_std"]
index_to_label = {
    0: "proliferation",
    1: "bean",
    2: "comma",
    3: "fold",
    4: "hatch",
    5: "death",
    6: "unfertilized",
}

DEVICE = "mps"
ZARR_GROUP_NAME = "dynamic_features"
IMGXY = (224, 224)

trained_model = SulstonNet.load_from_checkpoint(
    checkpoint_path,
    in_channels=len(channel_names),
    n_classes=len(index_to_label),
    index_to_label=index_to_label,
)

# Make sure the model is in eval mode for inference
trained_model.eval()

# Prepare your data
embryo_movie = Path(
    r"~/data/predict_development/celegans_embryos_dataset/230817_N2_heatshock/230817_2/"
    "embryo0.zarr"
)

device = torch.device(DEVICE)

# %% Add "raw" channel to the list of channels.

for i, ch in enumerate(channel_names):
    channel_movie = zarr.open(Path(embryo_movie, ZARR_GROUP_NAME, ch), mode="r")

    # First and last two frames are black. The array is in (T, C, H, W) shape.
    # We will treat T as a batch dimension for the clasification model.
    channel_movie = np.array(channel_movie[2:-2, ...])
    if i == 0:
        input_movie = channel_movie
    else:
        input_movie = np.concatenate((input_movie, channel_movie), axis=1)

input_tensor = torch.from_numpy(input_movie)  # Ignore the first channel (raw) that we added.
input_tensor = input_tensor.to(device)
input_tensor = torch.nn.functional.interpolate(input_tensor, size=IMGXY)

# Run inference
with torch.no_grad():
    logits = trained_model(input_tensor)
    predictions = torch.argmax(logits, axis=1)

predictions = predictions.to("cpu").numpy()

# Add raw_movie to input for display.
raw_movie = zarr.open(Path(embryo_movie, ZARR_GROUP_NAME, "raw"), mode="r")
raw_movie = np.array(raw_movie[2:-2, ...])
input_movie = np.concatenate((input_movie, raw_movie), axis=1)
# Show images with predictions overlaid.


viewer = napari.Viewer()
image_layer = viewer.add_image(
    input_movie,
    name=f"{embryo_movie.parent.name} / {embryo_movie.name}",
)


def text_overlay():
    i = viewer.dims.current_step[0]
    viewer.text_overlay.text = f"prediction:{index_to_label[predictions[i]]}"


# Connect the update function to the slider
viewer.dims.events.current_step.connect(text_overlay)
# Overlay settings
viewer.text_overlay.color = "white"
viewer.text_overlay.font_size = 18
viewer.text_overlay.visible = True

# Start the napari event loop
napari.run()
