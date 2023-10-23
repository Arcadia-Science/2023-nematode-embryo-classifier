# %% imports
from pathlib import Path
import nd2
import numpy as np
from iohub import open_ome_zarr
from tqdm import tqdm


nd2_path = (
    r"/mnt/embryostage-local/celegans_embryos_dataset/"
    "raw_data/100423/ChannelDIC_40x_Seq0012.nd2"
)
# We use the strain_condition_date_raw.zarr naming convention
ome_zarr_path = (
    r"/mnt/embryostage-local/celegans_embryos_dataset/"
    "raw_data/231004_TBD_control_raw.zarr"
)

channel_names = ["DIC40x"]


# %%


# @click.option("--nd2path", type=Path, help="Path to nd2 file")
# @click.option("--ome_zarr_path", type=Path, help="Path to output zarr")
# @click.command()
def nd2omezarr(nd2_path: str, ome_zarr_path: str) -> None:
    """
    Converts a nd2 file to an ome-zarr file.

    Args:
        nd2path (str): Path to the nd2 file.
        ome_zarr_path (str): Path to the output ome-zarr file.

    Returns:
        None
    """
    # Read the nd2 file as a dask array
    nd2dask = nd2.imread(nd2_path, dask=True)

    # Get the dimensions of the nd2 file
    T, P, Y, X = nd2dask.shape
    
    # Loop through each position in the nd2 file
    print(f"converting {nd2_path} to {ome_zarr_path}")
    
    for pos in tqdm(range(P)):
        # Get the dask array for the current position
        pos_dask = nd2dask[:, pos, :, :].rechunk(chunks=(T, Y, X))

        # Create a directory for the current position in the ome-zarr file
        pos_zarr_path = Path(ome_zarr_path, f"fov{pos}")
        pos_zarr_path.mkdir(parents=True, exist_ok=True)

        # Open the ome-zarr file for the current position
        pos_zarr = open_ome_zarr(pos_zarr_path, layout="fov", mode="w",
                                 channel_names=channel_names)

        # Convert the dask array to a numpy array and reshape it
        pos_array = np.asarray(pos_dask).reshape(T, 1, 1, Y, X)

        # Create an image in the ome-zarr file for the current position
        pos_zarr.create_image(data=pos_array, name="raw", chunks=(T, 1, 1, Y, X))

        # Close the ome-zarr file for the current position
        pos_zarr.close()


if __name__ == "__main__":
    nd2omezarr(nd2_path, ome_zarr_path)
