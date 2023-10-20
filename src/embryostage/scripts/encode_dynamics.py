# %% Import packages
from itertools import chain
from pathlib import Path
import napari
import zarr

from tqdm import tqdm

from embryostage.preprocess.utils import compute_morphodynamics

# %%
if __name__ == "__main__":
    database_path = "~/data/predict_development/celegans_embryos_dataset"
    strain = "N2"
    perturbation = "heatshock"
    date_stamp = "230817"
    FOVs = [str(x) for x in range(99)]
    pyramid_level = "0"

    all_embryos = Path(
        database_path,
        f"{date_stamp}_{strain}_{perturbation}",
    ).expanduser()

    for fov in FOVs:
        # Find all movies in an FOV.
        movie_paths = all_embryos.glob(f"{date_stamp}_{fov}/embryo*")

        movie_paths = list(movie_paths)
        if not movie_paths:
            print(
                f"No movie found at {date_stamp}_{strain}_{perturbation}"
                f"{date_stamp}_{fov}. Check the date stamp and FOV numbers."
            )
            continue

        print(f"Computing morphodynamics for FOV {fov}:")

        for movie_path in tqdm(movie_paths):
            # movie = open_ome_zarr(movie_path)[pyramid_level]
            movie = zarr.open(str(movie_path), mode="r")
            feature_imgs, features = compute_morphodynamics(movie)
            feature_dict = {features[n]: feature_imgs[:, n, ...] for n in range(len(features))}
            zarr.save_group(
                Path(movie_path, "dynamic_features"),
                **feature_dict,
            )
