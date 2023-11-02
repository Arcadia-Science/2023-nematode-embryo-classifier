import pathlib
import click
import monai.transforms as transforms
import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_model_summary import summary

from embryostage.cli import options as cli_options
from embryostage.metadata import get_annotations_filepath
from embryostage.models.classification import SulstonNet
from embryostage.models.data import EmbryoDataModule, EmbryoDataset


@cli_options.data_dirpath_option
@click.option(
    '--dataset-ids',
    type=str,
    help='The datasets to use for training, as a comma-separated list',
)
@click.option(
    '--logs-dirpath',
    type=pathlib.Path,
    help='The directory to which to write the training logs',
)
@click.command()
def main(data_dirpath, logs_dirpath, dataset_ids):
    '''
    train models for embryo classification using different combinations of channels
    '''

    random_seed = 2023
    pl.seed_everything(random_seed)

    channel_combinations = {
        "SulstonNet_raw": ["raw"],
        "SulstonNet_moving_mean_std": ["moving_mean", "moving_std"],
    }

    # transforms to use during training to augment the data
    transform = transforms.Compose(
        [
            transforms.RandFlip(prob=0.5, spatial_axis=(0, 1)),
            transforms.RandRotate(range_x=0.5 * np.pi, prob=0.2, padding_mode="border"),
            transforms.RandZoom(prob=0.2, min_zoom=0.8, max_zoom=1.2, padding_mode="edge"),
        ]
    )

    for experiment_name, channel_names in channel_combinations.items():
        print(f"Training model: {experiment_name}\n")

        embryo_dataset = EmbryoDataset(
            data_dirpath=(data_dirpath / 'encoded_dynamics'),
            channel_names=channel_names,
            annotations_csv=get_annotations_filepath(),
            dataset_ids=dataset_ids,
            transform=transform,
        )

        print(
            'Training with data from the following datasets: %s'
            % embryo_dataset.labels_df.dataset_id.unique()
        )

        embryo_data_module = EmbryoDataModule(
            dataset=embryo_dataset,
            transform=transform,
            batch_size=100,
            balance_classes=True,
        )
        embryo_data_module.setup()

        model = SulstonNet(
            n_input_channels=len(channel_names),
            n_classes=embryo_data_module.dataset.n_classes,
            index_to_label=embryo_data_module.dataset.index_to_label,
        )
        print(summary(model, model.example_input_array, show_hierarchical=False, max_depth=2))

        logger = pl.loggers.TensorBoardLogger(
            logs_dirpath, version=f"{experiment_name}", log_graph=True
        )

        # saves top-K checkpoints based on "val_loss" metric
        checkpoint_callback = ModelCheckpoint(
            save_top_k=5,
            monitor="val_loss",
            mode="min",
            filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
        )

        trainer = pl.Trainer(
            max_epochs=20, logger=logger, log_every_n_steps=10, callbacks=[checkpoint_callback]
        )

        trainer.fit(model, embryo_data_module)
        test_outputs = trainer.test(model, embryo_data_module.test_dataloader())
        print(test_outputs)


if __name__ == "__main__":
    main()
