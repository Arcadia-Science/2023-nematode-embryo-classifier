# %% imports
import threading  # threading is needed to run napari in nonblocking mode.

from pathlib import Path
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd

from embryostage.models.data import EmbryoDataModule, EmbryoDataset
from embryostage.preprocess.utils import compute_morphodynamics

dataset_path = Path("~/docs/data/predict_development/celegans_embryos_dataset").expanduser()
annotation_csv = Path(
    "~/docs/code/2023-celegans-sandbox/ground_truth/embryo_developmental_stage.csv"
).expanduser()

metadata_csv = Path(
    "~/docs/code/2023-celegans-sandbox/ground_truth/embryo_metadata.csv"
).expanduser()

channel_names = ["moving_mean", "moving_std"]


# %load_ext autoreload
# %autoreload 2
# # %% The script to explore all annotated embryos.
# We first create an EmbryoDataset object, which reads all annotations and annotated frames into memory.


# %% Test the dataset.

celegans_dataset = EmbryoDataset(dataset_path, channel_names, annotation_csv, metadata_csv)
sample_image, sample_class = celegans_dataset[42]

# Calculate number of embryos per stage
embryo_counts = celegans_dataset.labels_df.groupby("stage").count()
plt.bar(embryo_counts.index, embryo_counts["embryo_idx"])

# Create a new napari viewer
viewer = napari.Viewer()

# Add the dataset as an image layer
random_idx = np.random.randint(0, len(celegans_dataset), 500)
image_samples = np.empty((len(random_idx), *celegans_dataset[0][0].shape))
for i, data_idx in enumerate(random_idx):
    image_samples[i, :] = celegans_dataset[data_idx][0]
image_layer = viewer.add_image(image_samples, name="celegans_dataset")


# Function to update the text overlay
def text_overlay():
    i = viewer.dims.current_step[0]
    data_idx = random_idx[i]
    stage = celegans_dataset.labels_df.iloc[data_idx]["stage"]
    zarr_path = celegans_dataset.labels_df.iloc[data_idx]["zarr_path"]
    frame = celegans_dataset.labels_df.iloc[data_idx]["frame"]
    viewer.text_overlay.text = f"{stage}, {zarr_path}, {frame}"


# Connect the update function to the slider
viewer.dims.events.current_step.connect(text_overlay)
# Overlay settings
viewer.text_overlay.color = "white"
viewer.text_overlay.font_size = 18
viewer.text_overlay.visible = True

# Start the napari event loop
napari.run()

# %% Test the data loaders and class balancing - unbalanced.

embryo_data_module = EmbryoDataModule(
    dataset_path,
    channel_names,
    annotation_csv,
    metadata_csv,
    batch_size=100,
    balance_classes=False,
)
embryo_data_module.setup()

# Draw a batch of training data from biased dataset
train_loader_unblanaced = embryo_data_module.train_dataloader()

(
    batch_images_unbalanced,
    batch_labels_unbalanced,
) = next(iter(train_loader_unblanaced))


stages_unbalanced = [
    embryo_data_module.dataset.index_to_label[idx]
    for idx in batch_labels_unbalanced.numpy().astype(int)
]

stage_unbalanced_counts = {s: stages_unbalanced.count(s) for s in set(stages_unbalanced)}

plt.figure("Unbalanced dataset")
plt.bar(stage_unbalanced_counts.keys(), stage_unbalanced_counts.values())

# %% Test the data loaders and class balancing - balanced.

# Important to set the class balance flag when the DataModule is created.
embryo_data_module = EmbryoDataModule(
    dataset_path,
    channel_names,
    annotation_csv,
    metadata_csv,
    batch_size=100,
    balance_classes=True,
)
embryo_data_module.setup()

train_loader_balanced = embryo_data_module.train_dataloader()

(
    batch_images_balanced,
    batch_labels_balanced,
) = next(iter(train_loader_balanced))

stages_balanced = [
    embryo_data_module.dataset.index_to_label[idx]
    for idx in batch_labels_balanced.numpy().astype(int)
]
stage_balanced_counts = {s: stages_balanced.count(s) for s in set(stages_balanced)}

plt.figure("Balanced dataset")
plt.bar(stage_balanced_counts.keys(), stage_balanced_counts.values())

# %% Visualize batches in napari

# Create a new napari viewer
viewer = napari.Viewer()

# Add the batch images as an image layer
layer1 = viewer.add_image(batch_images_unbalanced.numpy(), name="batch_images_unbalanced")

# Add the batch images as an image layer
layer2 = viewer.add_image(batch_images_balanced.numpy(), name="batch_images_balanced")


# Function to update the text overlay
def text_overlay():
    i = viewer.dims.current_step[0]
    stage_b = stages_balanced[i]
    stage_un = stages_unbalanced[i]
    viewer.text_overlay.text = f"balanced: {stage_b}, unbalanced: {stage_un}"


# Connect the update function to the slider
viewer.dims.events.current_step.connect(text_overlay)

# Overlay settings
viewer.text_overlay.color = "white"
viewer.text_overlay.font_size = 18
viewer.text_overlay.visible = True

# Start the napari event loop
napari.run()
