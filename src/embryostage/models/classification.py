import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics

from torch import nn
from torchvision import models


class SulstonNet(pl.LightningModule):
    '''
    The SulstonNet model for embryo classification
    '''

    def __init__(self, n_input_channels=2, n_classes=5, index_to_label=None, xy_size=224):
        super(SulstonNet, self).__init__()
        self.n_classes = n_classes
        self.index_to_label = index_to_label
        self.n_input_channels = n_input_channels
        self.xy_size = xy_size

        # Initialize class names.
        if self.index_to_label is None:
            self.index_to_label = dict(
                zip(
                    list(range(self.n_classes)),
                    [str(i) for i in range(self.n_classes)],
                )
            )

        # Load pre-trained ResNet and adapt it.
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Update the model architecture to match
        # the number of input channels and output classes.
        conv1_out_channels = (
            self.resnet.conv1.out_channels
        )  # output channels from the first conv layer

        self.resnet.conv1 = nn.Conv2d(
            self.n_input_channels,
            conv1_out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=True,
        )  # replace the first conv layer

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(
            in_features, n_classes
        )  # update the last layer to output n_classes

        # Example input array
        self.example_input_array = torch.rand(
            1, self.n_input_channels, self.xy_size, self.xy_size
        )

        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )

        self.valid_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )

        self.test_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )

        self.train_cm = torchmetrics.classification.ConfusionMatrix(
            task="multiclass", num_classes=self.n_classes
        )

        self.val_cm = torchmetrics.classification.ConfusionMatrix(
            task="multiclass", num_classes=self.n_classes
        )

        self.test_cm = torchmetrics.classification.ConfusionMatrix(
            task="multiclass", num_classes=self.n_classes
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log("train_loss", loss)
        self.train_acc(logits, y)
        self.train_cm(logits, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log_predictions(x, logits, y, stage="train")

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log("val_loss", loss)

        self.valid_acc(logits, y)
        self.val_cm(logits, y)
        self.log("val_acc", self.valid_acc, on_step=False, on_epoch=True)
        self.log_predictions(x, logits, y, stage="val")

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        self.log("test_loss", loss)

        self.test_acc(logits, y)
        self.log("test_accuracy", self.test_acc)
        self.test_cm(logits, y)

        self.log_predictions(x, logits, y, stage="test")

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def on_train_epoch_end(self):
        # Log the confusion matrix at the end of the epoch
        confusion_matrix = self.train_cm.compute().cpu().numpy()
        self.logger.experiment.add_figure(
            "Train Confusion Matrix",
            plot_confusion_matrix(confusion_matrix, self.index_to_label),
            self.current_epoch,
        )
        self.train_cm.reset()

    def on_validation_epoch_end(self):
        # Log the confusion matrix at the end of the epoch
        confusion_matrix = self.val_cm.compute().cpu().numpy()

        self.logger.experiment.add_figure(
            "Validation Confusion Matrix",
            plot_confusion_matrix(confusion_matrix, self.index_to_label),
            self.current_epoch,
        )
        self.val_cm.reset()

    def on_test_epoch_end(self):
        # Log the confusion matrix at the end of the epoch
        confusion_matrix = self.test_cm.compute().cpu().numpy()

        self.logger.experiment.add_figure(
            "Test Confusion Matrix",
            plot_confusion_matrix(confusion_matrix, self.index_to_label),
            self.current_epoch,
        )

    def log_predictions(
        self,
        images,
        logits,
        targets,
        num_images=None,
        stage="training",
        log_every_n_steps=10,
    ):
        # Plot the images, predictions, and targets
        if self.global_step % log_every_n_steps != 0:
            return

        B, C, H, W = images.shape

        if num_images is None:
            num_images = self.n_classes

        with torch.no_grad():
            # Select 10 random images from the batch
            indices = torch.randperm(B)[:num_images]
            x_sample = images[indices].cpu().numpy()
            y_sample = targets[indices].cpu().numpy()
            logits_sample = logits[indices].cpu().numpy()

            output = np.argmax(logits_sample, axis=1)
            predictions = [self.index_to_label[p] for p in output]
            targets = [self.index_to_label[t] for t in y_sample]

        # Log the selected images with lightning using plot_predictions method
        fig, axs = plt.subplots(
            nrows=num_images // self.n_classes,
            ncols=self.n_classes,
            figsize=(10, 10),
        )
        for i in range(num_images):
            curr_sample = np.hstack(np.split(x_sample[i], C)).squeeze()
            axs[i].imshow(curr_sample)
            axs[i].set_title(f"pred: {predictions[i]} \n gt: {targets[i]}")
            axs[i].axis("off")
        plt.tight_layout()

        self.logger.experiment.add_figure(f"{stage}_samples", fig, self.global_step)

        return fig


def plot_confusion_matrix(confusion_matrix, index_to_label):
    # Create a figure and axis to plot the confusion matrix
    fig, ax = plt.subplots()

    # Create a color heatmap for the confusion matrix
    cax = ax.matshow(confusion_matrix, cmap="viridis")

    # Create a colorbar and set the label
    fig.colorbar(cax, label="Frequency")

    # Set labels for the classes

    ax.set_xticks(np.arange(len(index_to_label)))
    ax.set_yticks(np.arange(len(index_to_label)))
    ax.set_xticklabels(index_to_label.values(), rotation=45)
    ax.set_yticklabels(index_to_label.values())

    # Set labels for the axes
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Add text annotations to the confusion matrix
    for i in range(len(index_to_label)):
        for j in range(len(index_to_label)):
            ax.text(
                j,
                i,
                str(int(confusion_matrix[i, j])),
                ha="center",
                va="center",
                color="white",
            )

    return fig
