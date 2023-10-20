# %% imports
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import zarr

from tqdm import tqdm

from embryostage.models.classification import SulstonNet
from embryostage.preprocess.utils import get_movie_paths

# %%  Load a trained model from a checkpoint
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
device = torch.device(DEVICE)

# %% List of movies for which we should compute the inference.

# Prepare your data
database_path = "~/docs/data/predict_development/celegans_embryos_dataset"
strain = "N2"
perturbation = "heatshock"
date_stamp = "230817"
FOVs = range(99)
movie_paths = get_movie_paths(database_path, strain, perturbation, date_stamp, FOVs)


# %% Add "raw" channel to the list of channels.

print("Classifying movies...")

for idx_movie, movie_path in enumerate(tqdm(movie_paths)):
    for idx_ch, ch in enumerate(channel_names):
        channel_movie = zarr.open(Path(movie_path, ZARR_GROUP_NAME, ch), mode="r")

        # First and last two frames are black. The array is in (T, C, H, W) shape.
        # We will treat T as a batch dimension for the clasification model.
        channel_movie = np.array(channel_movie[2:-2, ...])
        if not idx_ch:
            input_movie = channel_movie
        else:
            input_movie = np.concatenate((input_movie, channel_movie), axis=1)

    input_tensor = torch.from_numpy(input_movie)
    input_tensor = input_tensor.to(device)
    input_tensor = torch.nn.functional.interpolate(input_tensor, size=IMGXY)

    # Run inference
    with torch.no_grad():
        logits = trained_model(input_tensor)
        labels = torch.argmax(logits, axis=1)
        labels = labels.to("cpu").numpy()
        labels = [index_to_label[label] for label in labels]

    labels_series = pd.Series(labels, name=f"{movie_path.parent.name}_{movie_path.name}")

    if not idx_movie:
        classifications_list = [labels_series]
    else:
        classifications_list = classifications_list + [labels_series]

classifications_df = pd.DataFrame(classifications_list)

# %% Save the predictions to a csv file.

csv_path = Path(
    database_path,
    f"{date_stamp}_{strain}_{perturbation}",
    "classifications.csv",
)

print(f"Saving predictions to {csv_path}")
classifications_df.to_csv(csv_path)

# %%
