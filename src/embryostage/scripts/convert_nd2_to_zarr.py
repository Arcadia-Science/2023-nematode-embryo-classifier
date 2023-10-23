# %% imports
from pathlib import Path
import click
import nd2
import numpy as np

from iohub import open_ome_zarr
from tqdm import tqdm


@click.option("--nd2-path", type=Path, help="Path to nd2 file")
@click.option("--ome-zarr-path", type=Path, help="Path to output zarr")
@click.command()
def convert_nd2_to_ome_zarr(nd2_path: str, ome_zarr_path: str) -> None:
    """
    Converts a nd2 file to an ome-zarr file.

    Args:
        nd2path (str): Path to the nd2 file.
        ome_zarr_path (str): Path to the output ome-zarr file.

    Returns:
        None
    """

    # TODO (KC): document where this channel name comes from
    channel_names = ["DIC40x"]

    # Read the nd2 file as a dask array
    nd2_dask = nd2.imread(nd2_path, dask=True)

    # Get the dimensions of the nd2 file
    num_timepoints, num_positions, size_y, size_x = nd2_dask.shape

    # Loop through each position in the nd2 file
    print(f"converting {nd2_path} to {ome_zarr_path}")

    for pos in tqdm(range(3)):
        # Get the dask array for the current position
        pos_dask = nd2_dask[:, pos, :, :].rechunk(chunks=(num_timepoints, size_y, size_x))

        # Create a directory for the current position in the ome-zarr file
        pos_zarr_path = Path(ome_zarr_path, f"fov{pos}")
        pos_zarr_path.mkdir(parents=True, exist_ok=True)

        # Open the ome-zarr file for the current position
        pos_zarr = open_ome_zarr(
            pos_zarr_path, layout="fov", mode="w", channel_names=channel_names
        )

        # Convert the dask array to a numpy array and reshape it
        pos_array = np.asarray(pos_dask).reshape(num_timepoints, 1, 1, size_y, size_x)

        # Create an image in the ome-zarr file for the current position
        pos_zarr.create_image(
            data=pos_array, name="raw", chunks=(num_positions, 1, 1, size_y, size_x)
        )

        # Close the ome-zarr file for the current position
        pos_zarr.close()


if __name__ == "__main__":
    convert_nd2_to_ome_zarr()
