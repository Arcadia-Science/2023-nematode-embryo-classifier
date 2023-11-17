from pathlib import Path
import numpy as np


def compute_morphodynamics(movie, features=None, t_window=5, normalize_features=True):
    if features is None:
        features = [
            "moving_std",
            "moving_mean",
            "raw",
        ]
    # Assuming movie is a T*XY numpy array
    T, _, _, Y, X = movie.shape
    edge_frames = int(t_window // 2)

    feature_imgs = np.random.random((T, len(features), 1, Y, X)).astype(np.float32)
    # Store data in the usual TCZYX format.

    for t_idx in range(edge_frames, T - edge_frames):
        for c_idx, channel in enumerate(features):
            if channel == "moving_std":
                feature_imgs[t_idx, c_idx, 0] = np.std(
                    movie[t_idx - edge_frames : t_idx + edge_frames],
                    axis=0,
                )
            elif channel == "moving_mean":
                feature_imgs[t_idx, c_idx, 0] = np.mean(
                    movie[t_idx - edge_frames : t_idx + edge_frames],
                    axis=0,
                )
            elif channel == "raw":
                feature_imgs[t_idx, c_idx, 0] = movie[t_idx]

    if normalize_features:
        t_range = range(edge_frames, T - edge_frames)

        for c_idx, channel in enumerate(features):
            flattened_features = feature_imgs[t_range, c_idx].flatten()
            max_val = np.max(flattened_features)
            min_val = np.min(flattened_features)

            feature_imgs[t_range, c_idx, 0] = (feature_imgs[t_range, c_idx, 0] - min_val) / (
                max_val - min_val
            )

    return feature_imgs, features


def get_cropped_embryo_filepaths(data_dirpath: Path, dataset_id: str, fov_ids: list[int]):
    '''
    get the paths to all embryo timelapses/movies for a given dataset and set of FOVs
    '''
    all_embryo_zarr_paths = []
    for fov_id in fov_ids:
        embryo_zarr_paths = list(
            (data_dirpath / 'cropped_embryos' / dataset_id / f'fov{fov_id}').glob(
                'embryo*.zarr'
            )
        )

        # embryo_zarr_paths will be an empty list if no directories match the glob pattern
        if not embryo_zarr_paths:
            print(f"Warning: no embryo movies found for {dataset_id} and FOV {fov_id}.")

        all_embryo_zarr_paths += embryo_zarr_paths

    return all_embryo_zarr_paths
