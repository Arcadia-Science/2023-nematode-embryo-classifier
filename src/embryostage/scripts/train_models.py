# %% imports
from embryostage.models.classification import SulstonNet
from embryostage.models.data import EmbryoDataModule
from pathlib import Path
import pytorch_lightning as pl
from pytorch_model_summary import summary
from pytorch_lightning.callbacks import ModelCheckpoint

RANDOM_SEED = 2023
pl.seed_everything(RANDOM_SEED)

# %load_ext autoreload
# %autoreload 2
# %% Construct the data module


dataset_path = Path(
    "~/docs/data/predict_development/celegans_embryos_dataset"
).expanduser()
annotation_csv = Path(
    "~/docs/code/2023-celegans-sandbox/ground_truth/embryo_developmental_stage.csv"
).expanduser()

metadata_csv = Path(
    "~/docs/code/2023-celegans-sandbox/ground_truth/embryo_metadata.csv"
).expanduser()

channel_combinations = {
    "sulstonNet_heatshock_7classes_raw": ["raw"],
    "sulstonNet_heatshock_7classes_flow": ["optical_flow"],
    "sulstonNet_heatshock_7classes_moving_mean_std": [
        "moving_mean",
        "moving_std",
    ],
}

for experiment_name, channel_names in channel_combinations.items():
    print(f"Training model: {experiment_name}\n_________________________")

    embryo_data_module = EmbryoDataModule(
        dataset_path,
        channel_names,
        annotation_csv,
        metadata_csv,
        batch_size=100,
        balance_classes=True,
    )
    embryo_data_module.setup()

    # %% Instantiate and examine the model
    model = SulstonNet(
        in_channels=len(channel_names),
        n_classes=embryo_data_module.dataset.n_classes,
        index_to_label=embryo_data_module.dataset.index_to_label,
    )

    # model_graph = draw_graph(
    #     model,
    #     model.example_input_array,
    #     directory=Path(dataset_path, "models"),
    #     graph_name=experiment_name,
    #     save_graph=True,
    # )

    print(
        summary(
            model,
            model.example_input_array,
            show_hierarchical=False,
            max_depth=2,
        )
    )

    # %% Train the model
    logger = pl.loggers.TensorBoardLogger(
        Path(dataset_path, "models"),
        version=f"{experiment_name}",
        log_graph=True,
    )

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = pl.Trainer(
        max_epochs=20,
        logger=logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model,
        embryo_data_module,
    )

    test_outputs = trainer.test(model, embryo_data_module.test_dataloader())
    print(test_outputs)
