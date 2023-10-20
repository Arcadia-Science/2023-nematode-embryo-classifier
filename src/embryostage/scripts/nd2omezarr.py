# %% imports
from pathlib import Path
import click
import nd2
import numpy as np

from iohub import open_ome_zarr

nd2path = r"D:\Matus\C_elegans\DQM327\230817\ChannelBF_20x_Seq0030.nd2"
ome_zarr_path = r"D:\Matus\C_elegans\predict_development\DQM327_heatshock_230817_test.zarr"


# %%


# @click.option("--nd2path", type=Path, help="Path to nd2 file")
# @click.option("--ome_zarr_path", type=Path, help="Path to output zarr")
# @click.command()
def nd2omezarr(nd2path, ome_zarr_path):
    nd2dask = nd2.imread(nd2path, dask=True)
    T, P, Y, X = nd2dask.shape

    for pos in range(P):
        pos_dask = nd2dask[:, pos, :, :].rechunk(chunks=(T, Y, X))
        pos_zarr_path = Path(ome_zarr_path, f"{pos}")
        pos_zarr_path.mkdir(parents=True, exist_ok=True)
        pos_zarr = open_ome_zarr(pos_zarr_path, layout="fov", mode="w", channel_names=["BF"])
        pos_array = np.asarray(pos_dask).reshape(T, 1, 1, Y, X)
        pos_zarr.create_image(data=pos_array, name="0", chunks=(1, 1, 1, Y, X))
        pos_zarr.close()
        print(f"Finished position {pos}")


if __name__ == "__main__":
    nd2omezarr(nd2path, ome_zarr_path)
