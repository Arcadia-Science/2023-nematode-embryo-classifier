import os

from pathlib import Path
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import zarr

from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from embryostage.models import constants


class EmbryoDataset(Dataset):
    def __init__(
        self, data_dirpath, channel_names, annotations_csv, dataset_ids, transform=None
    ):
        self.data_dirpath = Path(data_dirpath)

        # The length of this list indicates the number of channels
        # from which stage is predicted.
        # The names of channels describe what they represent
        # (e.g., moving_mean, moving_std, fluorescent_reporter, raw, optical_flow)
        self.channel_names = channel_names

        # The name of the group in the zarr store that contains the channels.
        self.channel_group = "dynamic_features"

        self.transform = transform

        # the x-y size of the cropped embryos
        self.xy_size = 224

        self.debug = False

        human_annotations = pd.read_csv(annotations_csv)
        self.labels_df = expand_annotations(human_annotations)

        if dataset_ids is not None:
            dataset_ids = [dataset_id.strip() for dataset_id in dataset_ids.split(",")]
            self.labels_df = self.labels_df.loc[
                self.labels_df["dataset_id"].astype(str).isin(dataset_ids)
            ]

        # the map from index to label (without labels that do not exist in the annotations)
        extant_labels = self.labels_df["stage"].unique()
        self.index_to_label = {
            index: label
            for index, label in constants.EMBRYO_STAGE_INDEX_TO_LABEL.items()
            if label in extant_labels
        }

        self.label_to_index = {label: index for index, label in self.index_to_label.items()}

        self.n_classes = len(self.index_to_label)

        self.label_to_code = {
            # CrossEntropyLoss expects logits.
            label: torch.nn.functional.one_hot(
                torch.tensor(index, dtype=torch.long), self.n_classes
            ).to(torch.float32)
            for label, index in self.label_to_index.items()
        }

    def resize_tensor(self, input_tensor, size):
        input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension
        input_tensor = torch.nn.functional.interpolate(
            input_tensor, size=size, mode="bilinear", align_corners=False
        )
        return input_tensor.squeeze(0)  # Remove the batch dimension

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Find the requested channels at a given index in dataset.
        zarr_path = self.data_dirpath / self.labels_df.loc[idx]["zarr_path"]
        frame = self.labels_df.loc[idx]["frame"]

        images = []
        for ch in self.channel_names:
            # Format of zarr store: zarr_path/dynamic_features_group/channel_name
            channel_path = zarr_path / self.channel_group / ch
            if os.path.exists(channel_path):
                images.append(zarr.open(channel_path, "r")[frame, :].squeeze())
            else:
                raise FileNotFoundError(f"No such file or directory: '{channel_path}'")

        images = torch.tensor(np.stack(images, axis=0))
        images = self.resize_tensor(images, (self.xy_size, self.xy_size))

        # One-hot encode the label and return.
        stage = self.labels_df.loc[idx]["stage"]
        label = self.label_to_index[stage]
        # We use CrossEntropyLoss, which expects target index
        # and not class probabilities.

        # The index row from annotation helps with debugging.
        index_dict = self.labels_df.loc[idx].to_dict()

        # The last tuple helps with debugging and ignored by training loop.
        return (images, label, index_dict) if self.debug else (images, label)

    def load_frames_into_memory(self):
        data = []

        pbar = tqdm(total=len(self.labels_df))

        for idx in range(len(self.labels_df)):
            frame = self.labels_df.iloc[idx]["frame"]
            zarr_path = self.labels_df.iloc[idx]["zarr_path"]

            pbar.set_description(f"loading frames from {zarr_path}")

            image_path = self.data_dirpath / zarr_path
            zarr_store = zarr.open(image_path, mode="r")
            image = zarr_store[frame].squeeze()
            if self.transform is not None:
                image = self.transform(image)
            data.append(image)

            pbar.update()

        return np.stack(data)


class EmbryoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        transform,
        split=(0.7, 0.15, 0.15),
        batch_size=32,
        balance_classes=True,
    ):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.split = split
        self.batch_size = batch_size
        self.balance_classes = balance_classes

    def setup(self, stage=None):
        '''
        note: `stage` is a required kwarg for lightning data modules.
        '''
        # Split the dataset into train, val, test sets.
        train_size = int(self.split[0] * len(self.dataset))
        val_size = int(self.split[1] * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = torch.utils.data.random_split(self.dataset, [train_size, val_size, test_size])

        # Create a weighted sampler to balance the classes
        # ------
        # Calculate the number of samples in each class.
        # Don't read labels as test_dataset[1], because the __getitem__ method loads images
        # too and is very slow when called for entire training dataset.

        train_labels_df = self.dataset.labels_df.iloc[self.train_dataset.indices]
        class_sample_count = train_labels_df.groupby("stage").count()["frame"]

        # Above returns a dictionary with stages as keys and frame counts as values.
        # Weights per class, this is also a dictionary.
        weights_class = 1.0 / class_sample_count

        # Weights per sample.
        samples_weights = [
            weights_class[train_labels_df.loc[i]["stage"]] for i in self.train_dataset.indices
        ]

        self.sampler = torch.utils.data.WeightedRandomSampler(
            samples_weights, len(samples_weights)
        )

    def collate_augmented(self, batch):
        transformed_batch = []
        for item in batch:
            images, label = item
            transformed_image = self.transform(images)
            transformed_batch.append((transformed_image, label))
        return default_collate(transformed_batch)

    def train_dataloader(self):
        if self.balance_classes:
            dataloader = DataLoader(
                self.train_dataset,
                self.batch_size,
                sampler=self.sampler,
                collate_fn=self.collate_augmented,
            )
        else:
            dataloader = DataLoader(
                self.train_dataset,
                self.batch_size,
                shuffle=True,
                collate_fn=self.collate_augmented,
            )

        return dataloader
        # shuffle during each epoch.

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size)


def expand_annotations_movie(df_movie, row, dev_stage_cols, t_start_cols, t_end_cols):
    for idx, col in enumerate(dev_stage_cols):
        t_start = row[t_start_cols[idx]]
        t_end = row[t_end_cols[idx]]
        current_stage = row[col]
        if not pd.isna(t_start) and not pd.isna(t_end) and not pd.isna(current_stage):
            t_start = int(t_start)
            t_end = int(t_end)
            for frame in range(t_start, t_end + 1):
                df_movie = pd.concat(
                    [
                        df_movie,
                        pd.DataFrame([{"frame": frame, "stage": current_stage}]),
                    ],
                    ignore_index=True,
                )
    return df_movie


def open_zarr_store(embryo_path):
    if not os.path.exists(embryo_path):
        print(f"{embryo_path} doesn't exist.")
        return None
    else:
        return zarr.open(str(embryo_path), mode="r")


def link_annotations_to_zarrs(data_dirpath: Path, annotations_path: Path, metadata_path: Path):
    """
    Links annotations of developmental stages to corresponding zarr stores for
    each embryo.

    Reads annotations of developmental stages from a CSV file and
    expands them to a dataframe with one row per time point.
    The dataframe is then saved as a CSV file in the corresponding zarr store.
    """

    annotations = pd.read_csv(annotations_path)
    annotations.astype({"datased_id": str, "fov_id": str}, copy=False)

    metadata = pd.read_csv(metadata_path)
    metadata.astype({"datased_id": str}, copy=False)

    # Find columns with annotations of developmental stages.
    dev_stage_cols = [col for col in annotations.columns if col.startswith("stage")]

    # Find columns with annotations of start times.
    t_start_cols = [col for col in annotations.columns if col.startswith("t_start")]

    # Find columns with annotations of end times.
    t_end_cols = [col for col in annotations.columns if col.startswith("t_end")]

    # Iterate over all embryos.
    pbar = tqdm(total=len(annotations))

    for _, row in annotations.iterrows():
        # the path to the zarr store for the cropped embryo corresponding to the current row
        embryo_path = (
            data_dirpath
            / row.dataset_id
            / f"fov{row.fov_id}"
            / f"embryo-{row.new_embryo_id}.zarr"
        )

        movie = open_zarr_store(embryo_path)
        if movie and movie.shape[0] > 1:
            df_movie = pd.DataFrame(columns=["frame", "stage"])
            df_movie = expand_annotations_movie(
                df_movie, row, dev_stage_cols, t_start_cols, t_end_cols
            )
            pbar.set_description(f"writing to {embryo_path.parent.name}/{embryo_path.name}")
            pbar.update()
            df_movie.to_csv(f"{embryo_path}/annotations.csv", index=False)


def expand_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Expand annotations of developmental stages, start times and end times
    into individual rows for each annotated frame of all movies of embryos.

    Parameters
    ----------
    annotations : pd.DataFrame
        A pandas DataFrame containing annotations of developmental stages,
        start times and end times - one row per movie.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing one row for each annotated frame of
        the movie, with columns for date, field of view (fov), embryo index,
        frame number, developmental stage and zarr path.
    """

    # Find columns with annotations of developmental stages.
    dev_stage_cols = [col for col in annotations.columns if col.startswith("stage")]

    # Find columns with annotations of start times.
    t_start_cols = [col for col in annotations.columns if col.startswith("t_start")]

    # Find columns with annotations of end times.
    t_end_cols = [col for col in annotations.columns if col.startswith("t_end")]

    expanded_annotations = pd.DataFrame(
        columns=[
            "dataset_id",
            "fov_id",
            "embryo_id",
            "frame",
            "stage",
            "zarr_path",
        ]
    )

    pbar = tqdm(total=len(annotations))
    for idx, row in annotations.iterrows():
        zarr_path = f"{row.dataset_id}/fov{row.fov_id}/embryo-{row.new_embryo_id}.zarr"

        # Iterate over all annotations of developmental stages.
        pbar.set_description(f"collecting annotations of {zarr_path}")
        pbar.update()
        for idx, col in enumerate(dev_stage_cols):
            t_start = row[t_start_cols[idx]]
            t_end = row[t_end_cols[idx]]
            current_stage = row[col]
            # Iterate over all time points at a given developmental
            # stage and expand the annotations.
            if not pd.isna(t_start) and not pd.isna(t_end) and not pd.isna(current_stage):
                t_start = int(t_start)
                t_end = int(t_end)
                # Write a row for each annotated frame of the movie.
                for frame in range(t_start, t_end + 1):
                    expanded_annotations = pd.concat(
                        [
                            expanded_annotations,
                            pd.DataFrame(
                                [
                                    {
                                        "dataset_id": row.dataset_id,
                                        "fov_id": row.fov_id,
                                        "embryo_id": row.new_embryo_id,
                                        "frame": frame,
                                        "stage": current_stage,
                                        "zarr_path": zarr_path,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )

    return expanded_annotations
