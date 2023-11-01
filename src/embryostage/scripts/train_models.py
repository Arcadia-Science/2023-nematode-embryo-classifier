import click
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_model_summary import summary

from embryostage.cli import options as cli_options
from embryostage.metadata import get_annotations_filepath, get_dataset_metadata_filepath
from embryostage.models.classification import SulstonNet
from embryostage.models.data import EmbryoDataModule


@cli_options.data_dirpath_option
@click.command()
def train_models(data_dirpath):
    '''
    train models for embryo classification using different combinations of channels
    '''

    random_seed = 2023
    pl.seed_everything(random_seed)

    channel_combinations = {
        "sulstonNet_heatshock_7classes_raw": ["raw"],
        "sulstonNet_heatshock_7classes_moving_mean_std": ["moving_mean", "moving_std"],
    }

    for experiment_name, channel_names in channel_combinations.items():
        print(f"Training model: {experiment_name}\n")

        embryo_data_module = EmbryoDataModule(
            data_dirpath,
            channel_names,
            annotations_csv=get_annotations_filepath(),
            metadata_csv=get_dataset_metadata_filepath(),
            batch_size=100,
            balance_classes=True,
        )
        embryo_data_module.setup()

        model = SulstonNet(
            in_channels=len(channel_names),
            n_classes=embryo_data_module.dataset.n_classes,
            index_to_label=embryo_data_module.dataset.index_to_label,
        )
        print(summary(model, model.example_input_array, show_hierarchical=False, max_depth=2))

        logger = pl.loggers.TensorBoardLogger(
            data_dirpath / "models", version=f"{experiment_name}", log_graph=True
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
    train_models()
